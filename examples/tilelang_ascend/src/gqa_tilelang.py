"""
GQA 解码注意力 (KV Cache) —— TileLang + tilelang-ascend 后端 (教学版)。

对应 docs/ops/06-gqa-kvcache.md: 解码一步 — 新 token 的 q (Hq, D) 对
KV Cache (Hkv, S, D) 打分 + 加权; GQA 分组 kv_head = hq // (Hq // Hkv):
    scores[hq, s] = ( q[hq] · K[kv, s] ) / sqrt(D)
    p[hq, :]      = softmax(scores[hq, :])
    out[hq, :]    = Σ_s p[hq, s] · V[kv, s, :]

实现 (写法约定同 rmsnorm/rope_tilelang): 每 program 负责一个 query 头,
3-pass 全程在 UB 内, 累加一律走 T.tile.add(acc, acc, tmp) 原位形式
(官方 rms_norm 示例验证过的写法):
    Pass 1  打分: 逐 s 拷 K 行 → mul q (逐元素) → T.reduce_sum
    Pass 2  softmax: T.reduce_max → broadcast sub → T.tile.exp → reduce_sum → div
    Pass 3  加权: 逐 s 拷 V 行 → ×p[s] (broadcast) → T.tile.add 原位累加
教学版为逐行串行 (S×D UB 操作/头), 不追求吞吐, 语义与 triton 在线版一致。
"""

import numpy as np

import tilelang
import tilelang.language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def gqa_decode(Hq: int, Hkv: int, S: int, D: int):
    """
    GQA 解码注意力 kernel: Q (Hq,D) fp16 + K/V cache (Hkv,S,D) fp16 → OUT (Hq,D) fp16.

    Args:
        Hq / Hkv: query 头数 / 共享 KV 头数 (Hq % Hkv == 0)
        S: cache 序列长度; D: head dim
    """
    assert Hq % Hkv == 0, f"Hq={Hq} 必须被 Hkv={Hkv} 整除"
    assert Hq % 2 == 0, f"Hq={Hq} 必须为偶数 (双 Vector 核拆分)"
    G = Hq // Hkv
    num_blocks = Hq // 2

    @T.prim_func
    def main(Q: T.Tensor((Hq, D), "float16"),
             K2: T.Tensor((Hkv * S, D), "float16"),   # (Hkv,S,D) 的 2D 视图 (内存同布局)
             V2: T.Tensor((Hkv * S, D), "float16"),
             OUT: T.Tensor((Hq, D), "float16")):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                hq = cid * 2 + vid
                kv = hq // G

                q_ub = T.alloc_ub((1, D), "float16")
                qf   = T.alloc_ub((1, D), "float32")
                k_ub = T.alloc_ub((1, D), "float16")
                kf   = T.alloc_ub((1, D), "float32")
                v_ub = T.alloc_ub((1, D), "float16")
                vf   = T.alloc_ub((1, D), "float32")
                prod = T.alloc_ub((1, D), "float32")
                acc  = T.alloc_ub((1, D), "float32")
                tmp  = T.alloc_ub((1, D), "float32")
                pb   = T.alloc_ub((1, D), "float32")
                out16 = T.alloc_ub((1, D), "float16")
                inv_sqrt_d = T.alloc_ub((1, 1), "float32")
                red  = T.alloc_ub((1, 1), "float32")
                mx   = T.alloc_ub((1, 1), "float32")
                lsum = T.alloc_ub((1, 1), "float32")
                p1   = T.alloc_ub((1, 1), "float32")
                diff = T.alloc_ub((1, 1), "float32")
                lb   = T.alloc_ub((1, D), "float32")
                row  = hq  # 语义标记

                # ---- inv_sqrt_d = 1/sqrt(D): 用 tile op 现算 (D 为编译期常量) ----
                T.tile.fill(inv_sqrt_d, 1.0)
                c_d = T.alloc_ub((1, 1), "float32")
                T.tile.fill(c_d, float(D))
                T.tile.sqrt(c_d, c_d)
                T.tile.div(inv_sqrt_d, inv_sqrt_d, c_d)

                # ---- q → UB, 升 fp32 ----
                T.copy(Q[hq, 0], q_ub)
                T.barrier_all()
                T.tile.cast(qf, q_ub, "CAST_NONE", D)

                # ---- Pass 1: 逐 s 打分求行 max (在线 max, 无切片写) ----
                T.tile.fill(mx, -T.infinity("float32"))
                for s in T.serial(S):
                    T.copy(K2[kv * S + s, 0], k_ub)
                    T.barrier_all()
                    T.tile.cast(kf, k_ub, "CAST_NONE", D)
                    T.tile.mul(prod, kf, qf)
                    T.reduce_sum(prod, red, dim=-1)
                    T.tile.mul(red, red, inv_sqrt_d)   # 1/sqrt(D) 经 (1,1) 标量 buffer
                    T.tile.max(mx, mx, red)            # 原位 max (官方 softmax 写法)

                # ---- Pass 2: 重算分数, exp(score - mx), 求和 l 并同时加权累加 ----
                T.tile.fill(acc, 0.0)
                T.tile.fill(lsum, 0.0)
                for s in T.serial(S):
                    T.copy(K2[kv * S + s, 0], k_ub)
                    T.barrier_all()
                    T.tile.cast(kf, k_ub, "CAST_NONE", D)
                    T.tile.mul(prod, kf, qf)
                    T.reduce_sum(prod, red, dim=-1)
                    T.tile.mul(red, red, inv_sqrt_d)
                    T.tile.sub(diff, red, mx)          # score - max
                    T.tile.exp(diff, diff)             # p[s] (数值稳定)
                    T.tile.add(lsum, lsum, diff)       # Σ p[s]
                    T.copy(V2[kv * S + s, 0], v_ub)
                    T.barrier_all()
                    T.tile.cast(vf, v_ub, "CAST_NONE", D)
                    T.tile.broadcast(pb, diff)         # p[s] 广播到 (1, D)
                    T.tile.mul(tmp, vf, pb)
                    T.tile.add(acc, acc, tmp)          # 原位累加 (官方推荐写法)

                # ---- 归一化: out = acc / l ----
                T.tile.broadcast(lb, lsum)
                T.tile.div(acc, acc, lb)
                T.tile.cast(out16, acc, "CAST_RINT", D)
                T.barrier_all()
                T.copy(out16, OUT[hq, 0])

    return main


def gqa_decode_tilelang(q, k_cache, v_cache):
    """
    便捷封装: numpy/torch → numpy。q (Hq,D) fp16, K/V (Hkv,S,D) fp16 → (Hq,D) fp16。
    """
    orig_dtype = np.dtype(np.float16)
    try:
        import torch
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
            k_cache = k_cache.detach().cpu().numpy()
            v_cache = v_cache.detach().cpu().numpy()
    except Exception:
        pass
    q = np.asarray(q)
    k_cache = np.asarray(k_cache)
    v_cache = np.asarray(v_cache)
    Hq, D = q.shape
    Hkv, S, _ = k_cache.shape
    assert Hq % 2 == 0, "教学版要求 Hq 为偶数 (双 Vector 核)"

    kernel = gqa_decode(Hq, Hkv, S, D)
    q_dev = torch.from_numpy(np.ascontiguousarray(q.astype(np.float16))).npu()
    k_dev = torch.from_numpy(np.ascontiguousarray(k_cache.astype(np.float16)).reshape(Hkv * S, D)).npu()
    v_dev = torch.from_numpy(np.ascontiguousarray(v_cache.astype(np.float16)).reshape(Hkv * S, D)).npu()
    out_dev = kernel(q_dev, k_dev, v_dev)
    torch.npu.synchronize()
    return out_dev.cpu().numpy().astype(orig_dtype)


# Ground truth (numpy), 与 examples/python/src/gqa.py 同公式
def gqa_decode_reference_numpy(q_np, k_np, v_np):
    q = np.asarray(q_np).astype(np.float32)
    Hq, D = q.shape
    Hkv, S, _ = k_np.shape
    G = Hq // Hkv
    kf = np.asarray(k_np).astype(np.float32)
    vf = np.asarray(v_np).astype(np.float32)
    qg = q.reshape(Hkv, G, D)
    scores = np.einsum("hgd,hsd->hgs", qg, kf) / np.sqrt(float(D))
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.einsum("hgs,hsd->hgd", p, vf)
    return out.reshape(Hq, D).astype(np.float16)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    Hq, Hkv, S, D = 4, 2, 128, 64
    q = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
    k = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
    v = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
    out = gqa_decode_tilelang(q, k, v)
    ref = gqa_decode_reference_numpy(q, k, v)
    err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-gqa smoke Hq={Hq} Hkv={Hkv} S={S} D={D}: max_abs_err={err:.6e}")
    assert err < 2e-2, f"tilelang gqa smoke FAIL, max_err={err}"
    print("tilelang-gqa smoke PASSED")
