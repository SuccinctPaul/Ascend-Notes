"""
INT8 对称量化 —— 纯 Python/NumPy 参考实现 (ground truth)。

对应 docs/ops/08-quantization.md §2.1 的对称量化 (per-row scale, §5.3:
逐行/逐块的 scale 比"整张一个 scale"好):

    scale = amax / 127                      # 每行一个缩放步长 (amax=max|x|)
    q     = clamp( round( x / scale ), -127, 127 )   # Quant
    x̂     = q * scale                       # Dequant

本文件提供:
  - quant_int8_reference(x)    -> (q_int8, scale_fp32)   # 逐行动态 absmax 量化
  - dequant_int8_reference(q, scale) -> x_fp16           # 反量化
  - roundtrip_max_error(x)     -> 逐行量化-反量化往返的最大绝对误差 (≤ scale)

约定: 输入 fp16 (Vector 主精度); amax/scale/round 在 fp32 里算 ("存窄算宽");
scale 用 fp32 存储并下发 (反量化精度敏感)。
"""

from __future__ import annotations

import numpy as np

QMAX = 127          # int8 对称量化的峰值 (去掉 -128, 保持对称)
EPS_AMIN = 1e-12    # 全零行的防除零下限


def quant_int8_reference(x: np.ndarray):
    """
    逐行 (最后一维) 动态 absmax 对称量化。

    Args:
        x: (..., D) fp16/fp32 ndarray
    Returns:
        q:     与 x 同 shape 的 int8
        scale: (...,) fp32, 每行一个步长 = amax/127
    """
    x = np.asarray(x)
    xf = x.astype(np.float32, copy=False)
    amax = np.max(np.abs(xf), axis=-1, keepdims=True)
    amax = np.maximum(amax, EPS_AMIN)            # 全零行防除零
    scale = (amax / QMAX).astype(np.float32)     # (..., 1)
    q = np.round(xf / scale)
    q = np.clip(q, -QMAX, QMAX).astype(np.int8)
    return q, scale.reshape(x.shape[:-1]).astype(np.float32)


def dequant_int8_reference(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """反量化: x̂ = q * scale, 输出 fp16 (与仓库 Vector 主精度一致)."""
    q = np.asarray(q).astype(np.float32)
    scale = np.asarray(scale, dtype=np.float32)[..., None]
    return (q * scale).astype(np.float16)


def roundtrip_max_error(x: np.ndarray) -> float:
    """量化→反量化往返的最大绝对误差 (理论上界 = 每行 scale, 典型 ≈ scale/2)."""
    q, scale = quant_int8_reference(x)
    xh = dequant_int8_reference(q, scale)
    return float(np.max(np.abs(xh.astype(np.float32) - x.astype(np.float32))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 自测 1: 往返误差上界 = 每行 scale
    for shape in [(16, 128), (64, 512), (8, 4096)]:
        x = (rng.standard_normal(shape) * 2.0).astype(np.float16)
        err = roundtrip_max_error(x)
        _, scale = quant_int8_reference(x)
        bound = float(np.max(scale))
        ok = err <= bound
        print(f"shape={str(shape):<12s} roundtrip_err={err:.6f} ≤ max_scale={bound:.6f} OK={ok}")
        assert ok
    # 自测 2: 值域 —— q 必落在 [-127, 127]
    x = (rng.standard_normal((32, 256)) * 100.0).astype(np.float16)
    q, _ = quant_int8_reference(x)
    print(f"q range = [{int(q.min())}, {int(q.max())}] (期望 [-127, 127])")
    assert q.min() >= -QMAX and q.max() <= QMAX
    # 自测 3: amax 元素量化后正好 = ±127
    x2 = rng.standard_normal((4, 64)).astype(np.float16)
    q2, _ = quant_int8_reference(x2)
    amax_idx = np.argmax(np.abs(x2.astype(np.float32)), axis=-1)
    hit = [abs(int(q2[r, amax_idx[r]])) == QMAX for r in range(4)]
    print(f"amax → ±127 hit = {hit}")
    assert all(hit)
    print("python-quant smoke tests PASSED")
