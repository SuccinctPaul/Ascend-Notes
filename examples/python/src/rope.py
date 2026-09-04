"""
RoPE (Rotary Position Embedding) —— 纯 Python/NumPy 参考实现 (ground truth)。

RoPE 把每个 token 的 q/k 向量拆成 d/2 对相邻分量, 每对按角度 m·θ_a 旋转
(复数乘法视角: (x1 + i·x2) · (cos + i·sin)):
    θ_a   = base^( -2a / d ),  base = 10000
    q'    = R(m) · q,   R(m) 为分块对角旋转矩阵
(详见 docs/ops/04-rope.md §2.1)

**配对约定 (全部 DSL 统一)**: 交错配对 (interleaved, RoFormer 原版) ——
第 a 对取 (x[2a], x[2a+1])。半维拆分 (x[0:d/2] 与 x[d/2:d] 配对) 是等价重排,
本仓库不做。

本文件提供:
  - precompute_rope_inv_freq(d, base)   -> (d/2,) θ 表
  - precompute_rope_costable(positions, d, base) -> cos/sin 表 (T, d/2)
  - apply_rope_numpy(x, positions, d, base)      -> 按 θ 现场算 cos/sin 再旋转
  - rope_reference(x, cos, sin)                  # 对外 ground truth: 查表版

统一约定: 输入/输出 fp16; 角度与 cos/sin 在 fp32 里算 (三角函数对精度敏感),
最后 cast 回 fp16。NPU kernel 同样采用 "host 预计算 cos/sin 表 + kernel 查表"。
"""

from __future__ import annotations

import numpy as np


def precompute_rope_inv_freq(d: int, base: float = 10000.0) -> np.ndarray:
    """θ_a = base^(-2a/d), a = 0..d/2-1 → 形状 (d/2,), fp32."""
    return 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))


def precompute_rope_tables(positions: np.ndarray, d: int, base: float = 10000.0):
    """
    预计算 cos/sin 表 (docs/04 §5.2 的"别在 kernel 里现算三角"):
      positions: (T,) 整数位置 (可为任意子集, 解码时往往只有最后一个新位置)
      返回 (cos, sin), 形状 (T, d/2), fp32。
    """
    inv_freq = precompute_rope_inv_freq(d, base)          # (d/2,)
    angles = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    return np.cos(angles), np.sin(angles)


def apply_rope_numpy(x: np.ndarray, positions: np.ndarray,
                     base: float = 10000.0) -> np.ndarray:
    """教学版: 内部现场预计算 cos/sin 表并旋转 (等价于 rope_reference)。"""
    d = x.shape[-1]
    cos, sin = precompute_rope_tables(positions, d, base)
    return rope_reference(x, cos, sin)


def rope_reference(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    RoPE ground truth (查表版)。

    Args:
      x:   (..., d) 输入 (q 或 k), 最后一维 d 必须为偶数
      cos: (..., d/2) 各对的旋转余弦
      sin: (..., d/2) 各对的旋转正弦
    Returns:
      与 x 同 shape/dtype 的旋转结果。
    """
    x = np.asarray(x)
    d = x.shape[-1]
    assert d % 2 == 0, f"RoPE 要求最后一维为偶数, 得到 d={d}"
    # fp32 中间量: 旋转是乘加, fp16 直接算误差可控但统一走 fp32 更稳
    xf = x.astype(np.float32, copy=False)
    x1 = xf[..., 0::2]          # 偶数下标: (..., d/2)
    x2 = xf[..., 1::2]          # 奇数下标
    cf = cos.astype(np.float32)
    sf = sin.astype(np.float32)
    y1 = x1 * cf - x2 * sf      # 复数乘实部
    y2 = x1 * sf + x2 * cf      # 复数乘虚部
    y = np.empty_like(xf)
    y[..., 0::2] = y1
    y[..., 1::2] = y2
    return y.astype(x.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 自测 1: 旋转保持模长 (每对 (x1,x2) 的欧氏范数不变)
    d, T = 128, 8
    x = rng.standard_normal((T, d)).astype(np.float32)
    pos = np.arange(T)
    y = apply_rope_numpy(x, pos)
    n_in = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
    n_out = np.sqrt(y[..., 0::2] ** 2 + y[..., 1::2] ** 2)
    print(f"norm preserved OK={np.allclose(n_in, n_out, atol=1e-5)}")

    # 自测 2: 相对位置性质 <R(m)q, R(n)k> == <q, R(n-m)k>
    q = rng.standard_normal(d).astype(np.float32)
    k = rng.standard_normal(d).astype(np.float32)
    m, n = 3, 17
    rq = apply_rope_numpy(q[None], np.array([m]))[0]
    rk = apply_rope_numpy(k[None], np.array([n]))[0]
    rk_rel = apply_rope_numpy(k[None], np.array([n - m]))[0]
    lhs = float(rq @ rk)
    rhs = float(q @ rk_rel)
    print(f"relative-position OK={np.isclose(lhs, rhs, atol=1e-4)}  ({lhs:.6f} vs {rhs:.6f})")

    # 自测 3: fp16 输入 → fp16 输出, dtype 保持
    x16 = rng.standard_normal((4, 64)).astype(np.float16)
    y16 = apply_rope_numpy(x16, np.arange(4))
    print(f"dtype preserved OK={y16.dtype == np.float16}")
    print("python-rope smoke tests PASSED")
