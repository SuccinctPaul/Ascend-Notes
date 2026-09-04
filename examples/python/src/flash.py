"""
FlashAttention 前向 —— 纯 Python/NumPy 参考实现 (ground truth)。

对应 docs/ops/07-flash-attention.md。FlashAttention 与标准注意力的**数学结果
完全一致** (online softmax 只是数值稳定的增量计算法), 因此 ground truth 就是
标准注意力:

    scores[h, m, s] = ( q[h,m] · k[h,s] ) / sqrt(D)
    p[h, m, :]      = softmax(scores[h, m, :])
    out[h, m, :]    = Σ_s p[h, m, s] · v[h, s, :]

Flash 的贡献在**实现层**: 分块 + online softmax 让中间分数 (L×S) 不落 HBM
(本参考实现为了清晰直接物化分数, 教学对比用)。

另提供 `flash_online_reference`: 严格按 flash 的 online softmax 逐步骤算
(块级 m/l/acc 增量), 用于验证 "online == 标准" 这一等价性本身。

约定: 输入/输出 fp16, 计算全程 fp32; 非因果 (bi-directional) 版本,
causal 作为扩展见 docs/07 §5。
"""

from __future__ import annotations

import numpy as np


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """标准注意力 ground truth: q/k/v (H, L, D) fp16 → out (H, L, D) fp16."""
    qf = np.asarray(q).astype(np.float32)
    kf = np.asarray(k).astype(np.float32)
    vf = np.asarray(v).astype(np.float32)
    D = qf.shape[-1]
    scores = np.einsum("hmd,hsd->hms", qf, kf) / np.sqrt(float(D))
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.einsum("hms,hsd->hmd", p, vf)
    return out.astype(np.float16)


def flash_online_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                           BLOCK_N: int = 64) -> np.ndarray:
    """
    Flash 风格 online softmax 参考实现 (逐 q 行, 逐 key 块的 m/l/acc 增量)。

    数学上与 attention_reference 完全等价; 用于验证 online 算法本身:
        m_new = max(m_old, max(scores_blk))
        l_new = l_old·exp(m_old - m_new) + Σ exp(scores_blk - m_new)
        acc   = acc·exp(m_old - m_new) + Σ exp(scores_blk - m_new) · v_blk
        out   = acc / l_final
    """
    qf = np.asarray(q).astype(np.float32)
    kf = np.asarray(k).astype(np.float32)
    vf = np.asarray(v).astype(np.float32)
    H, L, D = qf.shape
    S = kf.shape[1]
    scale = 1.0 / np.sqrt(float(D))
    out = np.zeros((H, L, D), dtype=np.float32)
    for h in range(H):
        for m in range(L):
            m_i, l_i = -np.inf, 0.0
            acc = np.zeros(D, dtype=np.float32)
            for s0 in range(0, S, BLOCK_N):
                sb = slice(s0, min(s0 + BLOCK_N, S))
                scores = kf[h, sb] @ qf[h, m] * scale
                m_new = max(m_i, float(scores.max()))
                a = np.exp(m_i - m_new) if np.isfinite(m_i) else 0.0
                p = np.exp(scores - m_new)
                l_i = l_i * a + p.sum()
                acc = acc * a + p @ vf[h, sb]
                m_i = m_new
            out[h, m] = acc / l_i
    return out.astype(np.float16)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, L, S, D = 2, 64, 128, 64
    q = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
    k = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
    v = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)

    ref = attention_reference(q, k, v)
    online = flash_online_reference(q, k, v, BLOCK_N=64)
    err = float(np.max(np.abs(ref.astype(np.float32) - online.astype(np.float32))))
    print(f"online-vs-standard max_err={err:.6e} (应 ≈ fp16 舍入级)")
    assert err < 5e-3

    # 数值稳定性: 大幅值输入 (score 量级 ~50) 不应爆炸
    q2 = (rng.standard_normal((H, L, D)) * 8.0).astype(np.float16)
    k2 = (rng.standard_normal((H, S, D)) * 8.0).astype(np.float16)
    v2 = (rng.standard_normal((H, S, D)) * 8.0).astype(np.float16)
    r2 = attention_reference(q2, k2, v2)
    print(f"large-magnitude finite={np.isfinite(r2).all()}")
    assert np.isfinite(r2).all()
    print("python-flash smoke tests PASSED")
