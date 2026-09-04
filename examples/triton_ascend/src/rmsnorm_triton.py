"""
RMSNorm kernel —— Triton on Ascend (triton-ascend 后端) 按行归一化实现。

    对每一行 x (hidden state):
        rms    = sqrt( (1/D) · Σ_j x_j² + eps )
        y[j]   = (x[j] / rms) · gamma[j]
    eps = 1e-6, gamma 为 per-dimension 可学习缩放 (shape (D,)).

实现策略 (教学版, 与 softmax_triton.py 同一骨架):
  1. 每个 program 负责一整行 (grid-stride 处理 rows > 65535).
  2. 对 D > BLOCK_SIZE 采用两阶段:
       - Pass A: 迭代 BLOCK_SIZE 子块, fp32 累加 Σx² (归约用宽精度, "存窄算宽").
       - Pass B: 先算 inv_rms = 1/sqrt(Σx²/D + eps) (标量, 只算一次),
                 再迭代子块做 y = x · inv_rms · gamma.
     关键手法: "乘倒数" 而非逐元素除 (docs/02 §5.2).
  3. 对 D <= BLOCK_SIZE, 两次 load 即可完成 (单 tile 驻留).

fp32 累加的必要性: fp16 尾数只有 ~11 位, d 上千时小平方项会被不断舍掉.
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, gamma_ptr, y_ptr, M, D, eps,
                   stride_xm, stride_ym,
                   BLOCK_SIZE: tl.constexpr):
    """
    行 RMSNorm: y = (x / rms(x)) * gamma, 对最后一维归一化.

    Args:
        x_ptr / y_ptr: 输入/输出 base pointer (fp16/fp32)
        gamma_ptr: 缩放参数 (D,)
        M: 行数, D: 特征维
        eps: 防除零小常数
        stride_xm / stride_ym: 行步长 (元素数)
        BLOCK_SIZE: 编译期 tile 大小
    """
    pid  = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    offs_d = tl.arange(0, BLOCK_SIZE)

    # Grid-stride: 每个 program 处理多行
    for row in range(pid, M, npid):
        x_row = row * stride_xm
        y_row = row * stride_ym

        # ==============================================================
        # Pass 1: fp32 累加 Σx² (跨 BLOCK_SIZE 子块)
        # ==============================================================
        sq_sum = 0.0
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx, mask=mask, other=0.0)
            xf = x_blk.to(tl.float32)
            sq_sum += tl.sum(xf * xf, axis=0)

        # ==============================================================
        # 标量: inv_rms = 1 / sqrt(Σx²/D + eps)  —— 每行只算一次
        # ==============================================================
        inv_rms = 1.0 / tl.math.sqrt(sq_sum / D + eps)

        # ==============================================================
        # Pass 2: y = x · inv_rms · gamma (fp32 中间, cast 回输入精度)
        # ==============================================================
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx, mask=mask, other=0.0)
            g_blk = tl.load(gamma_ptr + idx, mask=mask, other=0.0)
            yf = x_blk.to(tl.float32) * inv_rms * g_blk.to(tl.float32)
            tl.store(y_ptr + y_row + idx, yf.to(x_blk.dtype), mask=mask)


def rmsnorm_triton(x: torch.Tensor, gamma: torch.Tensor,
                   eps: float = 1e-6, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    对最后一维 (axis=-1) 做 RMSNorm.

    Args:
        x: torch.Tensor, shape (..., D), dtype=fp16|fp32, device=npu
        gamma: torch.Tensor, shape (D,), 同 dtype
        eps: 防除零常数, 默认 1e-6
        BLOCK_SIZE: 特征维分块大小, 默认 1024, 必须是 2^n.

    Returns:
        y: 和 x 同 shape/dtype/device
    """
    assert hasattr(x, "is_npu") and x.is_npu, \
        "triton-ascend kernel 仅支持 NPU device 张量"
    assert x.dtype in (torch.float16, torch.float32), \
        f"只支持 fp16/fp32, 但得到 {x.dtype}"
    assert gamma.numel() == x.shape[-1] and gamma.device == x.device

    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D
    flat = x.contiguous().view(M, D)
    gamma = gamma.contiguous()

    out = torch.empty_like(flat)

    grid = (min(65535, M),)
    rmsnorm_kernel[grid](
        flat, gamma, out, M, D, eps,
        flat.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(orig_shape)


# =============================================================================
# Ground truth (numpy) —— 与 examples/python/src/rmsnorm.py 同公式
# =============================================================================
def rmsnorm_reference_numpy(x_np: np.ndarray, gamma_np: np.ndarray,
                            eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    g = np.asarray(gamma_np).astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    y = x * inv_rms * g
    return y.astype(x_np.dtype, copy=False)


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
    g_dev = torch.rand(D, dtype=torch.float16, device="npu") + 0.5
    y_dev = rmsnorm_triton(x_dev, g_dev)

    ref = rmsnorm_reference_numpy(x_dev.cpu().numpy(), g_dev.cpu().numpy())
    y_np = y_dev.cpu().numpy()
    diff = np.max(np.abs(y_np.astype(np.float32) - ref.astype(np.float32)))
    print(f"triton-rmsnorm smoke M={M} D={D}: max_abs_err={diff:.6e}")

    # 归一化能量: y/gamma 每行均方 ≈ 1
    ms = np.mean(np.square(y_np.astype(np.float32) / g_dev.cpu().numpy().astype(np.float32)), axis=-1)
    print(f"  mean-square(y/gamma) range: [{float(ms.min()):.4f}, {float(ms.max()):.4f}]")
    assert diff < 5e-2, "triton rmsnorm failed basic smoke check"
    print("triton-rmsnorm smoke PASSED")
