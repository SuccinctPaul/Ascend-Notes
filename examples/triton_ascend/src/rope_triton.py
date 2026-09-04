"""
RoPE kernel —— Triton on Ascend (triton-ascend 后端)。

    对每个 token t 的向量 x 拆成 D/2 对, 按预计算的 cos/sin 表 (按 token 查表,
    docs/04 §5.2) 做二维旋转:
        y1 = x1·cos - x2·sin
        y2 = x1·sin + x2·cos

布局约定 (docs/04 §5.3 的两种排布, 本仓库两 种都用到):
  - **仓库统一对配约定是交错配对 (interleaved)**: pair_a = (x[2a], x[2a+1]),
    与 numpy/ascend_c/tilelang 版一致;
  - **kernel 内部用半维拆分 (half-split)**: 前半 x[0:D/2] 与后半 x[D/2:D] 配对。
    这是因为 triton-ascend 3.2.0 的 InterleaveOptimization pass 对 stride-2
    访存 (2*idx) 会触发编译器断言崩溃 (InterleaveStatusWithMaskOptimization),
    而成片的连续 load/store 对 Vector 单元也更友好 —— 正是 docs/04 §5.3 说的
    "半维拆分布局对成片访存最友好"。
  - 封装器 `rope_triton` 负责 interleaved ↔ half-split 的 layout 转换 (torch view),
    对外保持与参考实现一致的交错配对语义。

实现策略 (教学版):
  1. cos/sin 表由 host 预计算 (T × D/2), kernel 只做查表 + 乘加;
  2. 每个 program 负责一个 token 行 (grid-stride 处理 T > 65535);
  3. fp32 中间计算, cast 回输入精度; D/2 超过 BLOCK 时迭代子块。
"""

from __future__ import annotations

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, T, HALF_D,
                stride_xm, stride_ym, stride_cm,
                BLOCK: tl.constexpr):
    """
    半维拆分 RoPE: x 行布局为 [x1(D/2) | x2(D/2)], y 同布局。

    Args:
        x_ptr / y_ptr: 输入/输出 (T, 2*HALF_D) base pointer (fp16/fp32)
        cos_ptr / sin_ptr: 预计算表 (T, HALF_D), 与 x 同精度
        T: token 行数, HALF_D: 对数 (= D/2)
        stride_*: 行步长 (元素数)
        BLOCK: 编译期对数 tile 大小
    """
    pid  = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    offs = tl.arange(0, BLOCK)

    # Grid-stride: 每个 program 处理多个 token
    for t in range(pid, T, npid):
        for start in range(0, HALF_D, BLOCK):
            idx  = start + offs
            mask = idx < HALF_D
            # 半维拆分: 前半 / 后半各一次连续 load (无 stride-2 访存)
            x1 = tl.load(x_ptr + t * stride_xm + idx,           mask=mask, other=0.0)
            x2 = tl.load(x_ptr + t * stride_xm + HALF_D + idx,  mask=mask, other=0.0)
            c  = tl.load(cos_ptr + t * stride_cm + idx,         mask=mask, other=0.0)
            s  = tl.load(sin_ptr + t * stride_cm + idx,         mask=mask, other=0.0)
            # fp32 复数乘 (旋转)
            x1f = x1.to(tl.float32)
            x2f = x2.to(tl.float32)
            cf  = c.to(tl.float32)
            sf  = s.to(tl.float32)
            y1 = x1f * cf - x2f * sf
            y2 = x1f * sf + x2f * cf
            tl.store(y_ptr + t * stride_ym + idx,          y1.to(x1.dtype), mask=mask)
            tl.store(y_ptr + t * stride_ym + HALF_D + idx, y2.to(x1.dtype), mask=mask)


def precompute_rope_tables(positions: np.ndarray, d: int,
                           base: float = 10000.0):
    """
    host 侧预计算 cos/sin 表 (docs/04 §5.2: 别在 kernel 里现算三角).

    Returns:
        cos, sin: 形状 (T, d/2) 的 fp32 ndarray
    """
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    angles = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    return np.cos(angles), np.sin(angles)


def rope_triton(x: torch.Tensor, positions=None, base: float = 10000.0,
                BLOCK: int = 256, tables=None) -> torch.Tensor:
    """
    对最后一维做 RoPE 旋转 (对外是交错配对语义, kernel 内半维拆分).

    Args:
        x: torch.Tensor, shape (..., D) (D 为偶数), dtype=fp16|fp32, device=npu
        positions: 长度等于前导维合并后行数的整型序列 (ndarray / list / Tensor),
                   第 t 行的旋转角度为 positions[t] * θ_a; `tables` 给出时可省
        base: θ 底数, 默认 10000
        BLOCK: 对数 (D/2) 方向的 tile 大小
        tables: 可选 (cos_dev, sin_dev) —— 预先搬到 NPU 的 (T, D/2) 查表张量。
                **解码热路径应预构建一次并复用** (docs/04 §5.2/§5.6: 表常驻,
                只查新增行), 否则每次调用都会在 host 现算三角函数并 H2D,
                host 侧三角函数远贵于 NPU kernel 本身。

    Returns:
        y: 和 x 同 shape/dtype/device
    """
    assert hasattr(x, "is_npu") and x.is_npu, \
        "triton-ascend kernel 仅支持 NPU device 张量"
    assert x.dtype in (torch.float16, torch.float32), \
        f"只支持 fp16/fp32, 但得到 {x.dtype}"

    orig_shape = x.shape
    D = orig_shape[-1]
    assert D % 2 == 0, f"RoPE 要求最后一维为偶数, 得到 D={D}"
    HALF = D // 2
    T = x.numel() // D

    # ---- interleaved → half-split (纯 layout 转换, torch view 完成) ----
    # 交错: (x[0], x[1], x[2], x[3], ...) → 两半: (x[0], x[2], ...) | (x[1], x[3], ...)
    flat = x.contiguous().view(T, HALF, 2)
    x1 = flat[:, :, 0].contiguous()   # 偶数下标 (T, HALF)
    x2 = flat[:, :, 1].contiguous()   # 奇数下标 (T, HALF)
    xs = torch.cat([x1, x2], dim=1)   # (T, D): [前半 | 后半]

    # cos/sin 查表: 外部预构建复用, 否则 host 现算 (仅推荐离线/首次使用)
    if tables is not None:
        cos_dev, sin_dev = tables
        assert cos_dev.shape == (T, HALF) and cos_dev.is_npu and cos_dev.dtype == x.dtype
    else:
        assert positions is not None, "positions 与 tables 至少给一个"
        cos_np, sin_np = precompute_rope_tables(np.asarray(positions).reshape(-1), D, base)
        np_dtype = np.float16 if x.dtype == torch.float16 else np.float32
        cos_dev = torch.from_numpy(cos_np.astype(np_dtype)).npu()
        sin_dev = torch.from_numpy(sin_np.astype(np_dtype)).npu()

    ys = torch.empty_like(xs)

    grid = (min(65535, T),)
    rope_kernel[grid](
        xs, cos_dev, sin_dev, ys, T, HALF,
        xs.stride(0), ys.stride(0), cos_dev.stride(0),
        BLOCK=BLOCK,
    )

    # ---- half-split → interleaved ----
    y = torch.empty(T, HALF, 2, dtype=x.dtype, device=x.device)
    y[:, :, 0] = ys[:, :HALF]
    y[:, :, 1] = ys[:, HALF:]
    return y.view(orig_shape)


# =============================================================================
# Ground truth (numpy) —— 与 examples/python/src/rope.py 同公式 (交错配对)
# =============================================================================
def rope_reference_numpy(x_np: np.ndarray, positions: np.ndarray,
                         base: float = 10000.0) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    # positions 重塑为 x 的前导维 (支持 1D/2D/4D 输入)
    pos = np.asarray(positions, dtype=np.float32).reshape(x.shape[:-1])
    angles = pos[..., None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y = np.empty_like(x)
    y[..., 0::2] = x1 * cos - x2 * sin
    y[..., 1::2] = x1 * sin + x2 * cos
    return y.astype(x_np.dtype, copy=False)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("SKIP: torch.npu not available, triton-ascend kernel requires real NPU")
        sys.exit(0)

    T, D = 16, 128
    x_dev = (torch.randn((T, D), dtype=torch.float16, device="npu") * 2.0)
    y_dev = rope_triton(x_dev, np.arange(T))

    x_np = x_dev.cpu().numpy()
    ref = rope_reference_numpy(x_np, np.arange(T))
    y_np = y_dev.cpu().numpy()
    diff = np.max(np.abs(y_np.astype(np.float32) - ref.astype(np.float32)))
    print(f"triton-rope smoke T={T} D={D}: max_abs_err={diff:.6e}")

    # 旋转保范数 (每对)
    n_in = np.sqrt(x_np.astype(np.float32)[..., 0::2] ** 2 + x_np.astype(np.float32)[..., 1::2] ** 2)
    n_out = np.sqrt(y_np.astype(np.float32)[..., 0::2] ** 2 + y_np.astype(np.float32)[..., 1::2] ** 2)
    norm_err = float(np.max(np.abs(n_in - n_out)))
    print(f"  per-pair norm drift: {norm_err:.6e}")
    assert diff < 5e-2 and norm_err < 5e-2, "triton rope failed basic smoke check"
    print("triton-rope smoke PASSED")
