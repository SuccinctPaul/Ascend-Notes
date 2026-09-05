"""
GQA 解码注意力 (KV Cache) —— Triton on Ascend (triton-ascend 后端)。

对应 docs/ops/06-gqa-kvcache.md: 解码一步 — 新 token 的 q (Hq, D) 对
KV Cache (Hkv, S, D) 打分 + 加权; GQA 分组 kv_head = hq // (Hq // Hkv)。

实现 (在线 softmax / flash-decode 风格, 每 program 处理一个 query 头):
    对 s 沿 BLOCK_S 分块单趟扫描, 维护运行统计量 (无需物化 S 维分数):
        m_new = max(m_old, max(scores_blk))
        alpha = exp(m_old - m_new)
        l     = l·alpha + Σ exp(scores_blk - m_new)
        acc   = acc·alpha + Σ exp(scores_blk - m_new) · v_blk
    最终 out = acc / l
这就是 FlashAttention 的 online-softmax 核心 (docs/07 §5) 在解码形态的应用;
打分/加权全程 fp32, 输入/输出 fp16。
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def gqa_decode_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                      Hq, Hkv, S, D,
                      stride_qs, stride_kh, stride_ks, stride_vh, stride_vs,
                      stride_oh,
                      BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    每 program 负责一个 query 头的解码注意力 (online softmax, 单趟扫 KV Cache).

    Args:
        q_ptr: (Hq, D) fp16; k_ptr/v_ptr: (Hkv, S, D) fp16; out_ptr: (Hq, D) fp16
        stride_qs/stride_oh: q/out 的头步长; stride_kh/stride_vh: KV 头步长;
        stride_ks/stride_vs: KV 的 s 步长 (D 连续)
        BLOCK_S: s 方向 tile; BLOCK_D: D 方向 tile (≥ D, pad 0)
    """
    hq = tl.program_id(axis=0)
    kv = hq // (Hq // Hkv)                      # GQA 分组

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    scale = 1.0 / tl.sqrt(D.to(tl.float32))

    # q 行 → fp32
    q = tl.load(q_ptr + hq * stride_qs + offs_d, mask=d_mask, other=0.0)
    qf = q.to(tl.float32)

    # online softmax 运行统计量
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    for s0 in range(0, S, BLOCK_S):
        s_idx = s0 + offs_s
        s_mask = s_idx < S
        # K 块 (BLOCK_S, BLOCK_D) → 打分 (BLOCK_S,); 越界槽置 -inf
        k_blk = tl.load(k_ptr + kv * stride_kh + s_idx[:, None] * stride_ks + offs_d[None, :],
                        mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        kf = k_blk.to(tl.float32)
        scores = tl.sum(kf * qf[None, :], axis=1) * scale       # (BLOCK_S,)
        scores = tl.where(s_mask, scores, -float("inf"))

        # ---- online softmax 合并 (docs/07 §5 的核心三行) ----
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)       # 首块 m_i=-inf → alpha=0, acc/l 从 0 起步
        p = tl.exp(scores - m_new)        # 越界槽 (-inf) → exp=0, 不污染统计量
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # V 块加权累加
        v_blk = tl.load(v_ptr + kv * stride_vh + s_idx[:, None] * stride_vs + offs_d[None, :],
                        mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        vf = v_blk.to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vf, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + hq * stride_oh + offs_d, out.to(out_ptr.dtype.element_ty), mask=d_mask)


def gqa_decode_triton(q: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor,
                      BLOCK_S: int = 128) -> torch.Tensor:
    """
    GQA 解码注意力便捷封装。

    Args:
        q: (Hq, D) fp16, device=npu
        k_cache / v_cache: (Hkv, S, D) fp16, device=npu
    Returns:
        out: (Hq, D) fp16
    """
    assert hasattr(q, "is_npu") and q.is_npu
    Hq, D = q.shape
    Hkv, S, D2 = k_cache.shape
    assert D == D2 and v_cache.shape == k_cache.shape
    assert Hq % Hkv == 0
    assert q.dtype == torch.float16

    # D pad 到 2 的幂 (tl.arange 要求)
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    out = torch.empty((Hq, D), dtype=torch.float16, device=q.device)
    gqa_decode_kernel[(Hq,)](
        q, k_cache, v_cache, out,
        Hq, Hkv, S, D,
        q.stride(0), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        out.stride(0),
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    return out


# =============================================================================
# Ground truth (numpy) —— 与 examples/python/src/gqa.py 同公式
# =============================================================================
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


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)

    Hq, Hkv, S, D = 8, 2, 256, 128
    rng = np.random.default_rng(0)
    q_np = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
    k_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
    v_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)

    q_dev = torch.from_numpy(q_np).npu()
    k_dev = torch.from_numpy(k_np).npu()
    v_dev = torch.from_numpy(v_np).npu()
    out_dev = gqa_decode_triton(q_dev, k_dev, v_dev)
    out_np = out_dev.cpu().numpy()

    ref = gqa_decode_reference_numpy(q_np, k_np, v_np)
    err = float(np.max(np.abs(out_np.astype(np.float32) - ref.astype(np.float32))))
    print(f"triton-gqa smoke Hq={Hq} Hkv={Hkv} S={S} D={D}: max_abs_err={err:.6e}")
    assert err < 5e-2, "triton gqa failed smoke check"
    print("triton-gqa smoke PASSED")
