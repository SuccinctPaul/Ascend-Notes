"""
FlashAttention 前向 kernel —— Triton on Ascend (triton-ascend 后端)。

对应 docs/ops/07-flash-attention.md 的核心算法 (FA2, 非因果版):
每个 program 负责一个 (head, q 分块), 沿 K/V 的 s 维分块单趟扫描,
online softmax 增量维护 m/l/acc —— **L×S 的分数矩阵从头到尾不落 GM**:

    m_new = max(m_old, max(scores_blk))          (BLOCK_M 维行向量)
    alpha = exp(m_old - m_new)
    l     = l·alpha + Σ_n exp(scores - m_new)
    acc   = acc·alpha + p @ v_blk
    out   = acc / l

实现要点:
  - QK^T 与 PV 走 tl.dot (triton-ascend 自动映射 Cube, BLOCK 取 16 倍数);
  - K 由 wrapper 预转置为 (H, D, S) 传入 (torch 一次性布局变换), kernel 内
    全部成片连续访问, 不依赖 tl.trans —— 与 rope_triton 的布局约定同思路;
  - p 在 PV 前降 fp16 (Cube 原生精度), 累加器 acc 保持 fp32 (混合精度);
  - 越界 s 槽分数置 -inf, exp 后为 0, 不污染统计量。
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def flash_kernel(q_ptr, kt_ptr, v_ptr, out_ptr,
                 H, L, S, D,
                 stride_qh, stride_ql,
                 stride_kh, stride_kd,
                 stride_vh, stride_vs,
                 stride_oh, stride_ol,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    FA2 前向: q (H,L,D) fp16, kt (H,D,S) fp16 (K 的转置布局), v (H,S,D) fp16
    → out (H,L,D) fp16。1D grid = m块数 × H。
    """
    pid = tl.program_id(axis=0)
    num_m = tl.cdiv(L, BLOCK_M)
    pid_m = pid % num_m
    h = pid // num_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < L
    d_mask = offs_d < D

    # q tile (BLOCK_M, BLOCK_D) fp16 (dot 输入保持 Cube 原生精度)
    q = tl.load(q_ptr + h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    scale = 1.0 / tl.sqrt(D.to(tl.float32))
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for s0 in range(0, S, BLOCK_N):
        offs_n = s0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        # K^T tile (BLOCK_D, BLOCK_N) — kt 布局下连续成片
        kt = tl.load(kt_ptr + h * stride_kh + offs_d[:, None] * stride_kd + offs_n[None, :],
                     mask=d_mask[:, None] & n_mask[None, :], other=0.0)
        # scores (BLOCK_M, BLOCK_N), fp32 累加
        scores = tl.dot(q, kt) * scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # ---- online softmax 增量 (docs/07 §5 核心四行) ----
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)                # 首块 m_i=-inf → alpha=0
        p = tl.exp(scores - m_new[:, None])        # 越界槽 → 0
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_blk = tl.load(v_ptr + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :],
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_blk)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(out_ptr + h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :])


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
    """
    FlashAttention 前向便捷封装: q/k/v (H, L, S->?) 均 (H, L, D) / (H, S, D) fp16 NPU 张量。

    K 在 wrapper 内一次性转置为 (H, D, S) (kernel 走 K^T 布局, 免 tl.trans)。
    Returns: out (H, L, D) fp16
    """
    assert hasattr(q, "is_npu") and q.is_npu
    H, L, D = q.shape
    H2, S, D2 = k.shape
    assert H == H2 and D == D2 and v.shape == k.shape and q.dtype == torch.float16

    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    kt = k.transpose(1, 2).contiguous()          # (H, D, S) 一次性布局变换
    out = torch.empty((H, L, D), dtype=torch.float16, device=q.device)

    grid = (triton.cdiv(L, BLOCK_M) * H,)
    flash_kernel[grid](
        q, kt, v, out,
        H, L, S, D,
        q.stride(0), q.stride(1),
        kt.stride(0), kt.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


# =============================================================================
# Ground truth (numpy) —— 与 examples/python/src/flash.py 同公式
# =============================================================================
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


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)

    H, L, S, D = 2, 128, 256, 64
    rng = np.random.default_rng(0)
    q_np = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
    k_np = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
    v_np = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)

    out_dev = flash_attention_triton(torch.from_numpy(q_np).npu(),
                                     torch.from_numpy(np.ascontiguousarray(k_np)).npu(),
                                     torch.from_numpy(np.ascontiguousarray(v_np)).npu())
    out_np = out_dev.cpu().numpy()
    ref = attention_reference_numpy(q_np, k_np, v_np)
    err = float(np.max(np.abs(out_np.astype(np.float32) - ref.astype(np.float32))))
    print(f"triton-flash smoke H={H} L={L} S={S} D={D}: max_abs_err={err:.6e}")
    assert err < 5e-2, "triton flash failed smoke check"
    print("triton-flash smoke PASSED")
