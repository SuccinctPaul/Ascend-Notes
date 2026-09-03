"""
Softmax kernel —— Triton on Ascend (triton-ascend 后端) 按行 reduction 实现。

数值稳定的 softmax (按最后一维 axis=-1):
    For each row r:
        m   = max(x[r, :])
        e_j = exp(x[r, j] - m)
        s   = Σ_j e_j
        y[r, j] = e_j / s

实现策略 (教学版, 兼顾正确性 + 清晰):
  1. 每个 program 负责一整行 (grid-stride 处理 rows > 65535 的情况).
  2. 对 D > BLOCK_SIZE, 采用 "两阶段 row-wise accumulation":
       - Pass A: 迭代 BLOCK_SIZE 子块, 用 tl.maximum shift-register 算 row_max.
       - Pass B: 迭代子块, 写回 exp(x - row_max) 到输出 GM 暂存, 同时累加 sum_exp.
       - Pass C: 再次迭代子块, 读 exp(x-m) 暂存, 除以 sum_exp 并写回最终 y.
  3. 对 D <= BLOCK_SIZE, 直接单次 load / max / exp / sum / div / store.

同时, 也暴露一个 "单 BLOCK_SIZE tile + pad -inf" 的简化调用路径:
    softmax_triton(x, BLOCK_SIZE=1024)
若 D > BLOCK_SIZE, 会对最后一维 pad -inf 到 BLOCK_SIZE 的倍数 (exp(-inf)=0, 不影响
max/sum 结果, 算完再 unpad). 这样核内永远只处理 BLOCK_SIZE tile, 教学代码最清晰.
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


# =============================================================================
# Kernel: 每个 program 处理一整行; BLOCK_SIZE 是 D 维分块大小.
#         两阶段/三阶段迭代处理 D > BLOCK_SIZE 情况.
# =============================================================================
@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, D,
                   stride_xm, stride_xd,
                   stride_ym, stride_yd,
                   BLOCK_SIZE: tl.constexpr):
    """
    行 softmax: y = softmax(x, axis=-1)

    Args:
        x_ptr / y_ptr: base pointer (fp16/fp32)
        M: rows, D: columns
        stride_*: pointer 步长 (元素数)
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
        # Pass 1: 求 row_max (跨 BLOCK_SIZE 子块 shift-register 合并)
        # ==============================================================
        row_max = -float("inf")
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx * stride_xd,
                            mask=mask, other=-float("inf"))
            # x_blk 是 fp16 → 升 fp32 做 reduction
            cur_max = tl.max(x_blk.to(tl.float32), axis=0)
            row_max = tl.maximum(row_max, cur_max)

        # ==============================================================
        # Pass 2: 逐子块算 exp(x - row_max) 并写回 y 暂存, 同时累加 sum
        # ==============================================================
        sum_exp = 0.0
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            x_blk = tl.load(x_ptr + x_row + idx * stride_xd,
                            mask=mask, other=-float("inf"))
            xf = x_blk.to(tl.float32)
            shifted = xf - row_max
            exp_s   = tl.math.exp(shifted)
            # masked 位置 (D 外) 设 0 (exp(-inf)=0) 不影响 sum,
            # 但 tl.math.exp(-inf) 本就是 0, 这里直接累加即可.
            sum_exp += tl.sum(exp_s, axis=0)
            # 写回暂存 (用 mask 保证不越界)
            tl.store(y_ptr + y_row + idx * stride_yd,
                     exp_s.to(x_blk.dtype), mask=mask)

        # ==============================================================
        # Pass 3: 逐子块做 y = exp(x-m) / sum_exp
        # ==============================================================
        inv_sum = 1.0 / sum_exp
        for start in range(0, D, BLOCK_SIZE):
            idx  = start + offs_d
            mask = idx < D
            e_blk = tl.load(y_ptr + y_row + idx * stride_yd,
                            mask=mask, other=0.0)
            ef = e_blk.to(tl.float32)
            yf = ef * inv_sum
            tl.store(y_ptr + y_row + idx * stride_yd,
                     yf.to(e_blk.dtype), mask=mask)


# =============================================================================
# 用户 API: 支持任意形状 (..., D), 最后一维 softmax; 对 D 自动 pad -inf 到
#          BLOCK_SIZE 的倍数 (简化版), 再调用上述 kernel, 最后 unpad.
# =============================================================================
def softmax_triton(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    对最后一维 (axis=-1) 做数值稳定的 softmax.

    Args:
        x: torch.Tensor, shape (..., D), dtype=fp16|fp32, device=npu
        BLOCK_SIZE: 每行的 tile 分块大小, 默认 1024, 必须是 2^n.
                    若 D > BLOCK_SIZE, 先 pad 到下一个 BLOCK_SIZE 倍数.

    Returns:
        y: 和 x 同 shape/dtype/device
    """
    # --- sanity ---
    assert hasattr(x, "is_npu") and x.is_npu, \
        "triton-ascend kernel 仅支持 NPU device 张量"
    assert x.dtype in (torch.float16, torch.float32), \
        f"只支持 fp16/fp32, 但得到 {x.dtype}"

    # --- flatten 前导维度: shape -> (M, D) ---
    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D
    flat = x.contiguous().view(M, D)

    # --- 若 D > BLOCK_SIZE: pad -inf 到 BLOCK_SIZE 倍数 ---
    if D > BLOCK_SIZE:
        pad_len = BLOCK_SIZE - (D % BLOCK_SIZE)
        if pad_len == BLOCK_SIZE:
            pad_len = 0
        if pad_len > 0:
            flat_pad = torch.nn.functional.pad(
                flat.float() if x.dtype == torch.float32 else flat.half(),
                (0, pad_len), mode="constant",
                value=float("-inf")
            )
            D_pad = D + pad_len
        else:
            flat_pad = flat
            D_pad = D
    else:
        flat_pad = flat
        D_pad = D

    out_pad = torch.empty_like(flat_pad)

    # --- Launch grid: 1 program per row, limit <= 65535 (Ascend 限制) ---
    grid = (min(65535, M),)
    softmax_kernel[grid](
        flat_pad, out_pad, M, D_pad,
        flat_pad.stride(0), flat_pad.stride(1),
        out_pad.stride(0), out_pad.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # --- unpad ---
    if D_pad != D:
        out = out_pad[:, :D]
    else:
        out = out_pad

    return out.view(orig_shape)


# =============================================================================
# Ground truth (numpy)
# =============================================================================
def softmax_reference_numpy(x_np: np.ndarray, axis: int = -1) -> np.ndarray:
    """独立实现的 numpy softmax, 作为测试的 ground truth (数值稳定版)."""
    x = np.asarray(x_np).astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    y = e / s
    return y.astype(x_np.dtype, copy=False)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)

    M, D = 16, 128
    x_dev = (torch.randn((M, D), dtype=torch.float16, device="npu") * 3.0)
    y_dev = softmax_triton(x_dev)

    x_np = x_dev.cpu().numpy()
    ref  = softmax_reference_numpy(x_np, axis=-1)
    y_np = y_dev.cpu().numpy()

    diff = np.max(np.abs(y_np.astype(np.float32) - ref.astype(np.float32)))
    print(f"triton-softmax smoke M={M} D={D}: max_abs_err={diff:.6e}")

    # 额外检查: 每行和 ≈ 1
    row_sum_err = np.max(np.abs(y_np.astype(np.float32).sum(axis=-1) - 1.0))
    print(f"  row-sum max err: {row_sum_err:.6e}")
    assert diff < 5e-3, "triton softmax failed basic smoke check"
    print("triton-softmax smoke PASSED")
