"""
FlashAttention 前向 —— TileLang + tilelang-ascend 后端 (逐行 online softmax 教学版)。

对应 docs/ops/07-flash-attention.md: Flash 的算法内容 = 分块 + online softmax
(m/l/acc 增量, L×S 分数矩阵不落 GM)。本教学版对每个 query 行 (H×L 展开)
执行与官方 softmax 示例相同的 online 增量:
    逐 s: score → mx 原位 max → 重算 score → exp(score-mx) →
          lsum 原位加 → p[s] 广播 → acc 原位累加 (p·v)
最后 acc / lsum 归一化。**分数从不整体物化** —— 这就是 flash 的本质,
只是逐行串行 (教学地板), 无分块并行。

写法约定同 gqa_tilelang.py (2D cache 视图 + (1,1) 原位 tile 操作)。
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
def flash_attention(H: int, L: int, S: int, D: int):
    """
    FA 前向 kernel: Q (H*L, D) fp16 + K2/V2 (H*S, D) fp16 → OUT (H*L, D) fp16。

    Q/OUT 按 (H, L, D) 展平的行视图传入; K2/V2 是 (H, S, D) 的 2D 视图,
    第 h 个头的第 s 个 key 在行 h*S+s。
    """
    R = H * L
    assert R % 2 == 0, f"R=H*L={R} 必须为偶数 (双 Vector 核拆分)"
    num_blocks = R // 2

    @T.prim_func
    def main(Q: T.Tensor((R, D), "float16"),
             K2: T.Tensor((H * S, D), "float16"),
             V2: T.Tensor((H * S, D), "float16"),
             OUT: T.Tensor((R, D), "float16")):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                r = cid * 2 + vid
                h = r // L                       # 该 query 行所属的头

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
                lb   = T.alloc_ub((1, D), "float32")
                out16 = T.alloc_ub((1, D), "float16")
                inv_sqrt_d = T.alloc_ub((1, 1), "float32")
                red  = T.alloc_ub((1, 1), "float32")
                mx   = T.alloc_ub((1, 1), "float32")
                lsum = T.alloc_ub((1, 1), "float32")
                diff = T.alloc_ub((1, 1), "float32")

                # ---- inv_sqrt_d = 1/sqrt(D) (D 为编译期常量, tile op 现算) ----
                T.tile.fill(inv_sqrt_d, 1.0)
                c_d = T.alloc_ub((1, 1), "float32")
                T.tile.fill(c_d, float(D))
                T.tile.sqrt(c_d, c_d)
                T.tile.div(inv_sqrt_d, inv_sqrt_d, c_d)

                # ---- q 行 → UB, 升 fp32 ----
                T.copy(Q[r, 0], q_ub)
                T.barrier_all()
                T.tile.cast(qf, q_ub, "CAST_NONE", D)

                # ---- Pass 1: 逐 s 打分求行 max (online max, 不物化分数) ----
                T.tile.fill(mx, -T.infinity("float32"))
                for s in T.serial(S):
                    T.copy(K2[h * S + s, 0], k_ub)
                    T.barrier_all()
                    T.tile.cast(kf, k_ub, "CAST_NONE", D)
                    T.tile.mul(prod, kf, qf)
                    T.reduce_sum(prod, red, dim=-1)
                    T.tile.mul(red, red, inv_sqrt_d)
                    T.tile.max(mx, mx, red)

                # ---- Pass 2: 重算分数 → exp → l 与 acc 增量累加 ----
                T.tile.fill(acc, 0.0)
                T.tile.fill(lsum, 0.0)
                for s in T.serial(S):
                    T.copy(K2[h * S + s, 0], k_ub)
                    T.barrier_all()
                    T.tile.cast(kf, k_ub, "CAST_NONE", D)
                    T.tile.mul(prod, kf, qf)
                    T.reduce_sum(prod, red, dim=-1)
                    T.tile.mul(red, red, inv_sqrt_d)
                    T.tile.sub(diff, red, mx)
                    T.tile.exp(diff, diff)           # p[s]
                    T.tile.add(lsum, lsum, diff)
                    T.copy(V2[h * S + s, 0], v_ub)
                    T.barrier_all()
                    T.tile.cast(vf, v_ub, "CAST_NONE", D)
                    T.tile.broadcast(pb, diff)
                    T.tile.mul(tmp, vf, pb)
                    T.tile.add(acc, acc, tmp)

                # ---- 归一化 ----
                T.tile.broadcast(lb, lsum)
                T.tile.div(acc, acc, lb)
                T.tile.cast(out16, acc, "CAST_RINT", D)
                T.barrier_all()
                T.copy(out16, OUT[r, 0])

    return main


def flash_attention_tilelang(q, k, v):
    """便捷封装: q/k/v numpy/torch, q (H,L,D), k/v (H,S,D) → out (H,L,D) fp16."""
    try:
        import torch
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
            k = k.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
    except Exception:
        pass
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    H, L, D = q.shape
    H2, S, _ = k.shape
    assert H == H2

    kernel = flash_attention(H, L, S, D)
    q_dev = torch.from_numpy(np.ascontiguousarray(q.astype(np.float16)).reshape(H * L, D)).npu()
    k_dev = torch.from_numpy(np.ascontiguousarray(k.astype(np.float16)).reshape(H * S, D)).npu()
    v_dev = torch.from_numpy(np.ascontiguousarray(v.astype(np.float16)).reshape(H * S, D)).npu()
    out_dev = kernel(q_dev, k_dev, v_dev)
    torch.npu.synchronize()
    return out_dev.cpu().numpy().reshape(H, L, D)


# Ground truth (numpy), 与 examples/python/src/flash.py 同公式
def attention_reference_numpy(q_np, k_np, v_np):
    qf = np.asarray(q_np).astype(np.float32)
    kf = np.asarray(k_np).astype(np.float32)
    vf = np.asarray(v_np).astype(np.float32)
    D = qf.shape[-1]
    scores = np.einsum("hmd,hsd->hms", qf, kf) / np.sqrt(float(D))
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("hms,hsd->hmd", p, vf).astype(np.float16)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    H, L, S, D = 2, 8, 64, 64
    q = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
    k = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
    v = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
    out = flash_attention_tilelang(q, k, v)
    ref = attention_reference_numpy(q, k, v)
    err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-flash smoke H={H} L={L} S={S} D={D}: max_abs_err={err:.6e}")
    assert err < 2e-2, f"tilelang flash smoke FAIL, max_err={err}"
    print("tilelang-flash smoke PASSED")
