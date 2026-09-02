"""
GELU —— 纯 Python/NumPy 参考实现 (TBE 等价的"精确"语义基准)。

工业界实现全部采用 tanh 近似版 (见 ops/05-gelu.md §2.3):

    GELU(x) ≈ x · 0.5 · (1 + tanh( √(2/π) · (x + 0.044715 · x³) ))

本文件提供:
  - gelu_numpy(x: np.ndarray)      -> np.ndarray   # 最朴素, 按公式逐元素
  - gelu_scalar(x: float)          -> float        # 标量对照, 方便看懂公式
  - gelu_reference(x: np.ndarray)  -> np.ndarray   # 对外暴露的 ground truth,
                                                     其他 DSL 实现都与它对齐
这是所有其他 DSL (Ascend C / Triton-Ascend / TileLang-Ascend) 的正确性基准。
"""

from __future__ import annotations

import numpy as np

# GELU tanh 近似里的两个常数, 工业界统一使用。
_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2 / pi)
_CUBIC_COEF = 0.044715


def gelu_scalar(x: float) -> float:
    """标量版本, 一行把公式写清楚, 方便和文档/其他实现对照."""
    return float(x * 0.5 * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + _CUBIC_COEF * x * x * x))))


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """逐元素 tanh 近似 GELU. dtype 保持和输入一致."""
    x = np.asarray(x)
    inner = _SQRT_2_OVER_PI * (x + _CUBIC_COEF * np.power(x, 3))
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(x.dtype, copy=False)


# 对外 ground truth 名称统一, 其他文件 `from gelu import gelu_reference`.
gelu_reference = gelu_numpy


if __name__ == "__main__":
    # 自测: 几个典型输入, 结果值与 PyTorch nn.GELU(approximate='tanh') 对齐到 1e-6.
    samples = np.array([-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    print("x           =", samples)
    print("gelu_scalar =", [round(gelu_scalar(float(v)), 6) for v in samples])
    print("gelu_numpy  =", np.round(gelu_numpy(samples), 6))
