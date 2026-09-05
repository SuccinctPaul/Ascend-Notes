"""
GQA 解码注意力 (KV Cache) —— 纯 Python/NumPy 参考实现 (ground truth)。

对应 docs/ops/06-gqa-kvcache.md: 解码一步的注意力 —— 新 token 的 q 对
KV Cache 里全部历史 K/V 打分 + 加权求和; GQA 让 Hq 个 query 头分组共享
Hkv 个 K/V 头 (kv_head = hq // (Hq // Hkv)):

    scores[hq, s] = ( q[hq] · k_cache[hq//G, s] ) / sqrt(D)
    p[hq, :]      = softmax(scores[hq, :])
    out[hq, d]    = Σ_s p[hq, s] · v_cache[hq//G, s, d]

其中 G = Hq // Hkv (G=1 退化为 MQA, G=Hq 退化为 MHA)。

约定: 输入/输出 fp16; 打分/softmax/加权在 fp32 里做 ("存窄算宽");
softmax 数值稳定 (减 max)。
"""

from __future__ import annotations

import numpy as np


def gqa_decode_reference(q: np.ndarray, k_cache: np.ndarray,
                         v_cache: np.ndarray) -> np.ndarray:
    """
    GQA 解码注意力 ground truth。

    Args:
        q:       (Hq, D) fp16 — 新 token 的 query
        k_cache: (Hkv, S, D) fp16 — 历史 K 缓存
        v_cache: (Hkv, S, D) fp16 — 历史 V 缓存
    Returns:
        out: (Hq, D) fp16
    """
    q = np.asarray(q)
    Hq, D = q.shape
    Hkv, S, _ = np.asarray(k_cache).shape
    assert Hq % Hkv == 0, f"Hq={Hq} 必须被 Hkv={Hkv} 整除"
    G = Hq // Hkv

    qf = q.astype(np.float32)
    kf = np.asarray(k_cache).astype(np.float32)
    vf = np.asarray(v_cache).astype(np.float32)

    # 分组: (Hkv, G, D)
    qg = qf.reshape(Hkv, G, D)
    # 打分: (Hkv, G, S), 缩放 1/sqrt(D)
    scores = np.einsum("hgd,hsd->hgs", qg, kf) / np.sqrt(float(D))
    # 数值稳定 softmax (沿 s)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    # 加权求和: (Hkv, G, D)
    out = np.einsum("hgs,hsd->hgd", p, vf)
    return out.reshape(Hq, D).astype(np.float16)


def softmax_rows_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定行 softmax (独立暴露给 FA 等算子做交叉校验)."""
    xf = np.asarray(x).astype(np.float32)
    m = xf.max(axis=axis, keepdims=True)
    e = np.exp(xf - m)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 自测 1: MHA 退化 (Hkv == Hq) 应与逐头独立注意力一致
    H, S, D = 4, 32, 64
    q = rng.standard_normal((H, D)).astype(np.float16)
    k = rng.standard_normal((H, S, D)).astype(np.float16)
    v = rng.standard_normal((H, S, D)).astype(np.float16)
    out = gqa_decode_reference(q, k, v)
    # 手工逐头
    ref = np.empty((H, D), np.float32)
    for h in range(H):
        s = k[h].astype(np.float32) @ q[h].astype(np.float32) / np.sqrt(D)
        p = softmax_rows_reference(s)
        ref[h] = p @ v[h].astype(np.float32)
    err = float(np.max(np.abs(out.astype(np.float32) - ref)))
    print(f"MHA-fallback max_err={err:.6e}")
    assert err < 1e-2

    # 自测 2: MQA 退化 (Hkv == 1) 所有的头共享同一 K/V
    Hq = 8
    q8 = rng.standard_normal((Hq, D)).astype(np.float16)
    k1 = rng.standard_normal((1, S, D)).astype(np.float16)
    v1 = rng.standard_normal((1, S, D)).astype(np.float16)
    out8 = gqa_decode_reference(q8, k1, v1)
    assert out8.shape == (Hq, D)
    print(f"MQA-fallback shape OK {out8.shape}")

    # 自测 3: GQA 中间态 — 同组的两个 query 头消费同一份 K/V (输出不同但可复算)
    Hkv, G = 2, 4
    Hq = Hkv * G
    qg = rng.standard_normal((Hq, D)).astype(np.float16)
    kg = rng.standard_normal((Hkv, S, D)).astype(np.float16)
    vg = rng.standard_normal((Hkv, S, D)).astype(np.float16)
    outg = gqa_decode_reference(qg, kg, vg)
    # 手工算第 0 组
    s0 = kg[0].astype(np.float32) @ qg[0].astype(np.float32) / np.sqrt(D)
    p0 = softmax_rows_reference(s0)
    ref0 = (p0 @ vg[0].astype(np.float32)).astype(np.float16)
    err0 = float(np.max(np.abs(outg[0].astype(np.float32) - ref0.astype(np.float32))))
    print(f"GQA group-0 max_err={err0:.6e}")
    assert err0 < 1e-2
    print("python-gqa smoke tests PASSED")
