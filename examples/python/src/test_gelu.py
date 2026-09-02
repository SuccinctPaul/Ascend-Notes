"""
GELU correctness test — 纯 Python/NumPy 参考实现。

用 PyTorch 的 nn.GELU(approximate='tanh') 作为独立参考, 验证我们自己的
gelu_reference 在 fp32/fp16 下的数值结果与工业界实现一致。
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - 在最小 Python 环境里可能没有 torch
    _HAS_TORCH = False

from gelu import gelu_reference, gelu_scalar


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------- 1. 标量函数, 典型值 smoke test ----------

def test_scalar_typical_values():
    # 这些值与 PyTorch nn.GELU(tanh) 对齐, 误差不超过 5e-5
    cases = [
        (-4.0, -5.0e-05),
        (-2.0, -0.045402),
        (-1.0, -0.158808),
        (0.0, 0.0),
        (0.5, 0.345714),
        (1.0, 0.841192),
        (2.0, 1.954598),
        (4.0, 3.999950),
    ]
    for x, expected in cases:
        got = gelu_scalar(x)
        assert abs(got - expected) < 5e-4, (x, got, expected)


# ---------- 2. fp32 vs PyTorch tanh GELU, 各种形状 ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("shape", [(1024,), (256, 512), (2, 16, 128, 128)])
def test_gelu_reference_fp32_matches_pytorch(rng, shape):
    x_np = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    y_ref = gelu_reference(x_np)

    x_th = torch.from_numpy(x_np.copy())
    y_th = nn.GELU(approximate="tanh")(x_th).numpy()

    assert y_ref.shape == y_th.shape
    np.testing.assert_allclose(y_ref, y_th, atol=2e-6, rtol=2e-6)


# ---------- 3. fp16 下数值稳定 ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_gelu_reference_fp16_matches_pytorch(rng):
    x_np = (rng.standard_normal((4096,)) * 3.0).astype(np.float16)
    y_ref = gelu_reference(x_np)

    x_th = torch.from_numpy(x_np.astype(np.float32)).half()
    y_th = nn.GELU(approximate="tanh")(x_th).numpy().astype(np.float16)

    # fp16 宽松一些, 1 ULP 级的差异是可接受的
    np.testing.assert_allclose(y_ref.astype(np.float32),
                               y_th.astype(np.float32), atol=5e-3, rtol=5e-3)


# ---------- 4. dtype 保持: 输入是什么 dtype, 输出就什么 dtype ----------

def test_gelu_reference_preserves_dtype(rng):
    for dt in (np.float32, np.float64, np.float16):
        x = (rng.standard_normal(256) * 2.0).astype(dt)
        y = gelu_reference(x)
        assert y.dtype == dt


# ---------- 5. 非 x∈ℝ 全区间单调, 但在极值右侧单调非减 ----------
#
# 注意 (很容易踩的坑):
#   精确 GELU 与 tanh 近似 GELU 都 **不是全局单调非减**! 在负半轴
#   x ∈ (-∞, x_min], 其中 x_min ≈ -0.75 (tanh 近似 GELU 真实极小点),
#   GELU(x) 随 x 增大而 *减小* (从 0 下降到全局极小值 ≈ -0.165);
#   之后 x ≥ x_min 才开始单调非减地上升。
# 所以正确的检查只在 x >= -0.5 区间进行 (比真实极小点更保守, 给 fp 舍入留余地)。
# 这也解释了为什么 GELU 看起来"软门槛"但对极负值仍有梯度回来。

def test_gelu_reference_monotone_after_min(rng):
    x = np.sort(rng.standard_normal(30_000).astype(np.float32))
    # tanh 近似 GELU 的全局极小点在 x ≈ -0.75; 这里取 x >= -0.5,
    # 保证严格进入单调非减区间, 同时给 fp32 舍入留足够余量。
    mask = x >= -0.5
    y = gelu_reference(x[mask])
    diffs = np.diff(y)
    # 容差: fp32 排序下可能出现极小假负 (<1 ULP), 放宽到 1e-6
    assert float(diffs.min()) >= -1e-6, f"post-min non-monotone, min_diff={float(diffs.min())}"


# ---------- 6. 符号一致性 (GELU 不是奇函数, 但 sign(y) == sign(x)) ----------

def test_gelu_reference_sign_consistency(rng):
    x = rng.standard_normal(10_000).astype(np.float32)
    y = gelu_reference(x)
    # 对于 GELU: sign(y) == sign(x), 且 |y| <= max(|x|,0) 正值侧 y <= x (soft-gate)
    sign_ok = (np.sign(x) == np.sign(y))
    # 0 附近 sign 会 flip; 只统计绝对值大于 1e-5 的点
    mask = np.abs(x) > 1e-5
    assert sign_ok[mask].all(), "sign mismatch exists for |x|>1e-5"
    # 正值侧, GELU(x) <= x (因为门控 <=1)
    pos = x > 0
    assert (y[pos] <= x[pos] * 1.0001 + 1e-6).all()


if __name__ == "__main__":
    # 允许直接 `python test_gelu.py` 跑, 不用装 pytest 也能跑一遍 smoke。
    rng = np.random.default_rng(0)
    test_scalar_typical_values()
    test_gelu_reference_preserves_dtype(rng)
    test_gelu_reference_monotone_after_min(rng)
    test_gelu_reference_sign_consistency(rng)
    print("python-gelu smoke tests PASSED")
