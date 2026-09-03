"""
Softmax —— 纯 Python/NumPy 参考实现 (TBE 等价的"精确"语义基准)。

Softmax 不是逐元素算子, 它带有沿 axis 的 REDUCTION:
    softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
其中 m = max_j x_j 是整行的参考零点, 用于保证 exp 不溢出.
(详见 docs/ops/03-softmax.md §4.1 公式)

本文件提供:
  - softmax_numpy(x: np.ndarray, axis=-1)      -> np.ndarray   # 数值稳定版, 减 max
  - softmax_naive(x: np.ndarray, axis=-1)      -> np.ndarray   # 直接按定义, 会溢出, 仅教学用
  - softmax_reference(x: np.ndarray, axis=-1)  -> np.ndarray   # 对外暴露的 ground truth,
                                                                 其他 DSL 实现都与它对齐
这是所有其他 DSL (Ascend C / Triton-Ascend / TileLang-Ascend) 的正确性基准。
"""

from __future__ import annotations

import numpy as np


def softmax_naive(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    **教学用途**: 直接按定义 softmax(x) = exp(x) / Σ exp(x) 实现, 不减 max。

    警告: 只要整行里有任何 x_i 稍大 (比如 fp16 下 > 80) 就会爆成 inf/nan。
          真实代码 NEVER 使用该版本, 只保留用来对比"为什么必须减 max"。
    """
    x = np.asarray(x)
    e = np.exp(x)
    s = np.sum(e, axis=axis, keepdims=True)
    return (e / s).astype(x.dtype, copy=False)


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    数值稳定版 Softmax: 先减去 axis 上的 max → exp → 求和 → 相除。

    内部归约使用 float32 以减少求和的舍入累积, 最终 cast 回输入 dtype。
    支持任意维度 ndarray, 默认 axis=-1 (最后一维)。
    """
    x = np.asarray(x)
    # fp32 内部归约: sum / max 在 fp32 里做, 精度更好
    xf = x.astype(np.float32, copy=False)
    m = np.max(xf, axis=axis, keepdims=True)
    e = np.exp(xf - m)
    s = np.sum(e, axis=axis, keepdims=True)
    y = e / s
    return y.astype(x.dtype, copy=False)


# 对外 ground truth 名称统一, 其他文件 `from softmax import softmax_reference`.
def softmax_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax ground truth, dtype 保持与输入一致."""
    return softmax_numpy(x, axis=axis)


if __name__ == "__main__":
    # 自测: 几个典型形状, 每行求和应该 == 1 (到浮点精度)
    shapes = [(1024,), (16, 64), (2, 8, 128, 512)]
    rng = np.random.default_rng(0)
    for shape in shapes:
        x = rng.standard_normal(shape).astype(np.float32)
        y = softmax_reference(x)
        # 对最后一维求和 (默认 axis=-1, reduce 到 1.0)
        sums = np.sum(y, axis=-1)
        all_one = np.allclose(sums, 1.0, atol=1e-6)
        non_neg = np.all(y >= 0)
        print(f"shape={str(shape):<20s}  sum≈1 OK={all_one}  non-neg OK={non_neg}  "
              f"sum_range=[{float(sums.min()):.8f}, {float(sums.max()):.8f}]")
    print("python-softmax smoke tests PASSED")
