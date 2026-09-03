"""
GELU kernel —— Triton on Ascend (triton-ascend 后端) 逐元素实现。

和 gemm_triton.py 不同, GELU 是**逐元素算子**: 没有 reduction, 天然 embarrassingly
parallel。一个 program 负责 BLOCK_SIZE 个元素, 每个 AI Core 跑多个 program。

公式 (与仓库其他 GELU 实现完全对齐, 数值一致):

    GELU(x) = x * 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

硬件映射 (triton-ascend 后端自动完成):
  - tl.load/store   → GM ↔ UB 数据搬运 (Vector 单元)
  - tl.math.tanh    → Vector 单元 tanh 近似指令 (或多项式展开, 视后端而定)
  - 标量乘/加       → Vector 单元 MAC (128 或 256 元素/拍, 与 SoC 相关)
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


# ---- tanh 近似 GELU 的常数. 由于 triton kernel 内无法直接引用 module-level 普通全局,
# 这里存一份 tl.constexpr 版本供核内使用 (核外 numpy 参考仍用普通 Python float).
_SQRT_2_OVER_PI = 0.7978845608028654
_CUBIC_COEF     = 0.044715
_HALF           = 0.5
_ONE            = 1.0


@triton.jit
def gelu_kernel(x_ptr, y_ptr, N,
                SQRT2_OVER_PI: tl.constexpr,
                CUBIC_COEF:    tl.constexpr,
                HALF:          tl.constexpr,
                ONE:           tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    """
    每个 program 按 grid-stride 循环处理 BLOCK_SIZE × K 个元素, 支持:
      - 任意 N (含 N 不整除 BLOCK_SIZE, 或 grid 数 < ceil(N/BLOCK))
      - Ascend 1D grid <= 65535 的 runtime 限制 (否则 coreDim=65536 invalid)

    使用方式:
        grid = lambda opt: (min(65535, triton.cdiv(N, BLOCK_SIZE)),)
        gelu_kernel[grid](x, y, N, ...)
    当 N / BLOCK_SIZE > 65535 (例如 N=128M, BLOCK=1024 → 128K programs)
    时, kernel 内部用 pid + pid + num_programs 跳步, 让每个 program 多次
    循环以覆盖完整个 N.
    """
    pid  = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)   # 实际 grid 大小

    base = pid * BLOCK_SIZE
    step = npid * BLOCK_SIZE        # 下一轮 pid 跳步

    offs = tl.arange(0, BLOCK_SIZE)
    while base < N:
        idx  = base + offs
        mask = idx < N

        # --- 1. GM → UB: 载入 BLOCK_SIZE 个 fp16 ---
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)   # tl.float16

        # --- 2. 计算内联: 升 fp32 做中间运算 ---
        xf    = x.to(tl.float32)
        x3    = xf * xf * xf
        inner = SQRT2_OVER_PI * (xf + CUBIC_COEF * x3)
        t     = tl.math.tanh(inner)
        y     = xf * HALF * (ONE + t)

        # --- 3. UB → GM: 写回 fp16 ---
        tl.store(y_ptr + idx, y.to(tl.float16), mask=mask)

        base += step


def gelu_triton(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    便捷封装: 任意形状 fp16/fp32 张量, flatten → kernel → 还原 shape.

    Args:
        x: torch.Tensor, 任意 shape, dtype=fp16|fp32, device 必须是 npu
        block_size: 每个 program 处理的元素数, 建议 512/1024/2048, 必须是 2^n

    Returns:
        y: 和 x 同 shape / dtype / device
    """
    # --- sanity: 需要在 npu 上 ---
    assert hasattr(x, "is_npu") and x.is_npu, "triton-ascend kernel 仅支持 NPU device 张量"
    assert x.dtype in (torch.float16, torch.float32), f"只支持 fp16/fp32, 但得到 {x.dtype}"

    # --- 1D 化 ---
    flat = x.contiguous().view(-1)
    N = flat.numel()
    y_flat = torch.empty_like(flat)

    # kernel 内部统一写死 fp16 读/写以匹配最常用场景; 对 fp32 输入/输出走 cast 包装
    if x.dtype == torch.float32:
        flat_h = flat.to(torch.float16)
        y_h = torch.empty_like(flat_h)
    else:
        flat_h = flat
        y_h = y_flat

    grid = (min(65535, triton.cdiv(N, block_size)),)
    gelu_kernel[grid](
        flat_h, y_h, N,
        _SQRT_2_OVER_PI, _CUBIC_COEF, _HALF, _ONE,
        BLOCK_SIZE=block_size,
    )

    if x.dtype == torch.float32:
        y_flat = y_h.to(torch.float32)

    return y_flat.view_as(x)


# ---------------------- ground truth (和 examples/python/src/gelu.py 同公式) ----------------------

def gelu_reference_numpy(x_np: np.ndarray) -> np.ndarray:
    """独立实现的 numpy GELU, 作为 pytest 的 ground truth。"""
    x = np.asarray(x_np).astype(np.float32)
    inner = _SQRT_2_OVER_PI * (x + _CUBIC_COEF * np.power(x, 3))
    y = 0.5 * x * (1.0 + np.tanh(inner))
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    # smoke: 只在有 npu 环境时跑
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)
    x = torch.randn(4096, dtype=torch.float16, device="npu") * 3.0
    y = gelu_triton(x)
    ref = gelu_reference_numpy(x.cpu().numpy())
    diff = np.max(np.abs(y.cpu().numpy().astype(np.float32) - ref.astype(np.float32)))
    print(f"triton-gelu smoke: max_abs_err={diff:.6f}")
    assert diff < 5e-3, "triton gelu failed basic smoke check"
    print("triton-gelu smoke PASSED")
