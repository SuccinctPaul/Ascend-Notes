"""
INT8 对称量化 kernel —— Triton on Ascend (triton-ascend 后端)。

对应 docs/ops/08-quantization.md §2.1 的对称量化 (per-row scale, §5.3):

    Pass 1 (quant):  scale[r] = max|x[r,:]| / 127            (fp32 归约)
    Pass 2 (quant):  q[r,c]  = clamp(round(x[r,c]/scale[r])) → int8
    dequant:         x̂[r,c]  = q[r,c] * scale[r]             (Vector 反量化, §5.4)

实现策略 (与 rmsnorm_triton 同骨架):
  1. 每个 program 负责一行 (grid-stride);
  2. D > BLOCK_SIZE 时迭代子块 (absmax shift-register 合并);
  3. absmax/round 在 fp32 里算; scale 用 fp32 存储 (反量化精度敏感);
  4. 由于 scale 取自同一行的 absmax, x/scale ∈ [-127, 127] 自然满足,
     round 后无需再 clamp (仍保留 clamp 语句作防御, 语义与 docs 公式一致)。
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl

QMAX = 127.0


@triton.jit
def quant_kernel(x_ptr, q_ptr, scale_ptr, M, D,
                 stride_xm, stride_qm,
                 BLOCK_SIZE: tl.constexpr):
    """逐行 absmax 对称量化: x (M, D) fp16/fp32 → q (M, D) int8 + scale (M,) fp32."""
    pid  = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    offs_d = tl.arange(0, BLOCK_SIZE)

    for row in range(pid, M, npid):
        x_row = row * stride_xm
        q_row = row * stride_qm

        # ---- Pass 1: fp32 归约 absmax (跨子块 shift-register 合并) ----
        amax = 0.0
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx, mask=mask, other=0.0)
            cur = tl.max(tl.abs(x_blk.to(tl.float32)), axis=0)
            amax = tl.maximum(amax, cur)

        # ---- scale = amax/127 (防除零), fp32 下发 ----
        scale = tl.maximum(amax, 1e-12) / 127.0

        # ---- Pass 2: q = clamp(round(x/scale)) → int8 ----
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx, mask=mask, other=0.0)
            qf = x_blk.to(tl.float32) / scale
            qf = tl.minimum(tl.maximum(tl.floor(qf + 0.5), -127.0), 127.0)  # round + 防御性 clamp
            tl.store(q_ptr + q_row + idx, qf.to(tl.int8), mask=mask)

        tl.store(scale_ptr + row, scale)


@triton.jit
def dequant_kernel(q_ptr, scale_ptr, y_ptr, M, D,
                   stride_qm, stride_ym,
                   BLOCK_SIZE: tl.constexpr):
    """反量化: y = q(int8) * scale(fp32, 按行广播) → 与 y_dtype 同精度."""
    pid  = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    offs_d = tl.arange(0, BLOCK_SIZE)

    for row in range(pid, M, npid):
        scale = tl.load(scale_ptr + row)
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            q_blk = tl.load(q_ptr + row * stride_qm + idx, mask=mask, other=0)
            yf = q_blk.to(tl.float32) * scale
            tl.store(y_ptr + row * stride_ym + idx,
                     yf.to(y_ptr.dtype.element_ty), mask=mask)


def quant_int8_triton(x: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    逐行动态 absmax 对称量化。

    Args:
        x: (..., D) fp16|fp32, device=npu
    Returns:
        q: (..., D) int8
        scale: (...,) fp32
    """
    assert hasattr(x, "is_npu") and x.is_npu
    assert x.dtype in (torch.float16, torch.float32)
    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D
    flat = x.contiguous().view(M, D)

    q = torch.empty((M, D), dtype=torch.int8, device=x.device)
    scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    grid = (min(65535, M),)
    quant_kernel[grid](flat, q, scale, M, D,
                       flat.stride(0), q.stride(0),
                       BLOCK_SIZE=BLOCK_SIZE)
    return q.view(orig_shape), scale


def dequant_int8_triton(q: torch.Tensor, scale: torch.Tensor,
                        out_dtype: torch.dtype = torch.float16,
                        BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """反量化: y = q * scale (逐行广播), 输出 out_dtype (默认 fp16)."""
    assert hasattr(q, "is_npu") and q.is_npu and q.dtype == torch.int8
    orig_shape = q.shape
    D = orig_shape[-1]
    M = q.numel() // D
    flat = q.contiguous().view(M, D)
    scale = scale.contiguous().view(M)

    y = torch.empty((M, D), dtype=out_dtype, device=q.device)
    grid = (min(65535, M),)
    dequant_kernel[grid](flat, scale, y, M, D,
                         flat.stride(0), y.stride(0),
                         BLOCK_SIZE=BLOCK_SIZE)
    return y.view(orig_shape)


# =============================================================================
# Ground truth (numpy) —— 与 examples/python/src/quant.py 同公式
# =============================================================================
def quant_int8_reference_numpy(x_np: np.ndarray):
    x = np.asarray(x_np).astype(np.float32)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    amax = np.maximum(amax, 1e-12)
    scale = (amax / QMAX).astype(np.float32)
    q = np.clip(np.round(x / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale.reshape(x.shape[:-1])


def dequant_int8_reference_numpy(q_np: np.ndarray, scale_np: np.ndarray) -> np.ndarray:
    qf = np.asarray(q_np).astype(np.float32)
    return (qf * np.asarray(scale_np, dtype=np.float32)[..., None]).astype(np.float16)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)

    M, D = 16, 512
    x_dev = (torch.randn((M, D), dtype=torch.float16, device="npu") * 2.0)
    q_dev, s_dev = quant_int8_triton(x_dev)
    y_dev = dequant_int8_triton(q_dev, s_dev, torch.float16)

    x_np = x_dev.cpu().numpy()
    q_ref, s_ref = quant_int8_reference_numpy(x_np)
    y_ref = dequant_int8_reference_numpy(q_ref, s_ref)

    y_np = y_dev.cpu().numpy()
    q_np = q_dev.cpu().numpy()
    s_np = s_dev.cpu().numpy()
    q_match = float(np.mean(q_np.astype(np.int32) == q_ref.astype(np.int32)))
    err = float(np.max(np.abs(y_np.astype(np.float32) - y_ref.astype(np.float32))))
    rt_err = float(np.max(np.abs(y_np.astype(np.float32) - x_np.astype(np.float32))))
    scale_err = float(np.max(np.abs(s_np - s_ref)))
    print(f"triton-quant smoke M={M} D={D}:")
    print(f"  q 元素一致率 = {q_match:.4f} (round-half 语义差允许少量 ±1)")
    print(f"  scale max_err = {scale_err:.3e}")
    print(f"  dequant vs ref max_err = {err:.3e}")
    print(f"  roundtrip max_err = {rt_err:.6f} (上界 = max scale = {float(s_ref.max()):.6f})")
    assert q_match > 0.999 and err <= float(s_ref.max()) + 1e-6 and rt_err <= float(s_ref.max()) + 1e-6
    print("triton-quant smoke PASSED")
