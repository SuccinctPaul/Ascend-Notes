"""
Softmax correctness test — 纯 Python/NumPy 参考实现。

用 PyTorch 的 torch.nn.functional.softmax 作为独立参考, 验证我们自己的
softmax_reference 在 fp32/fp16 下的数值结果与工业界实现一致, 以及行和=1、
非负、dtype 保持、数值稳定性 vs 朴素版等性质。
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - 在最小 Python 环境里可能没有 torch
    _HAS_TORCH = False

from softmax import softmax_reference, softmax_naive


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------- 1. 每行求和 == 1.0 ----------

@pytest.mark.parametrize("shape", [(128,), (16, 64), (2, 4, 32, 128)])
def test_row_sum_equals_one(rng, shape):
    x = rng.standard_normal(shape).astype(np.float32)
    y = softmax_reference(x)
    sums = np.sum(y, axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# ---------- 2. 所有输出元素非负 ----------

@pytest.mark.parametrize("shape", [(256,), (8, 32), (2, 2, 16, 256)])
def test_non_negative(rng, shape):
    x = rng.standard_normal(shape).astype(np.float32)
    y = softmax_reference(x)
    assert np.all(y >= 0.0), "softmax 输出应全部 >= 0"


# ---------- 3. fp32 vs PyTorch F.softmax ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("shape", [(1024,), (32, 64), (2, 8, 16, 128)])
def test_fp32_matches_torch_softmax(rng, shape):
    x_np = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    y_ref = softmax_reference(x_np)

    x_th = torch.from_numpy(x_np.copy())
    y_th = F.softmax(x_th, dim=-1).numpy()

    assert y_ref.shape == y_th.shape
    np.testing.assert_allclose(y_ref, y_th, atol=1e-5, rtol=1e-5)


# ---------- 4. fp16 vs PyTorch F.softmax ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fp16_matches_torch_softmax(rng):
    # 用多个形状跑一遍 fp16, 容差放宽
    shapes = [(256,), (16, 64), (2, 4, 32, 128)]
    for shape in shapes:
        x_np = (rng.standard_normal(shape) * 3.0).astype(np.float16)
        y_ref = softmax_reference(x_np)

        x_th = torch.from_numpy(x_np.astype(np.float32)).half()
        y_th = F.softmax(x_th, dim=-1).numpy().astype(np.float16)

        # fp16 舍入累加会有一定误差, 给宽松容差
        np.testing.assert_allclose(y_ref.astype(np.float32),
                                   y_th.astype(np.float32), atol=1e-2, rtol=1e-2)


# ---------- 5. dtype 保持 ----------

@pytest.mark.parametrize("dt", [np.float32, np.float64, np.float16])
def test_preserves_dtype(rng, dt):
    x = (rng.standard_normal((256,)) * 2.0).astype(dt)
    y = softmax_reference(x)
    assert y.dtype == dt, f"输入 {dt}, 输出应为 {dt}, 实际 {y.dtype}"


# ---------- 6. 数值稳定性 vs 朴素版 (减 max 的必要性) ----------

def test_numerical_stability_vs_naive(rng):
    # 构造大值输入: 均匀分布 [0, 100]
    x = rng.uniform(low=0.0, high=100.0, size=(4, 64)).astype(np.float32)

    y_ref = softmax_reference(x)
    assert np.isfinite(y_ref).all(), "参考版 (减 max) 应输出全为有限值"

    y_naive = softmax_naive(x)
    # 朴素版在 x 量级 ~100 时 exp(x) 会溢出 → 出现 inf / nan / 非 1.0 sum
    has_overflow = (not np.isfinite(y_naive).all()) or not np.allclose(
        np.nansum(np.where(np.isfinite(y_naive), y_naive, 0.0), axis=-1), 1.0, atol=1e-3)
    assert has_overflow, "softmax_naive 在大值输入下应该出现 inf/nan/归一化失败"


# ---------- 7. 梯度方向 sanity: 同一行里输入越大, softmax 输出越大 ----------

def test_gradient_direction(rng):
    # 构造同一行里严格递增的输入 → softmax 输出也应保持严格递增
    N = 32
    # 每行 32 个元素, 按顺序从 -3 到 3 线性排开 (同一行内严格递增)
    base = np.linspace(-3.0, 3.0, N, dtype=np.float32)
    x_rows = np.stack([base + float(i) * 0.1 for i in range(8)], axis=0)  # (8, 32)
    y = softmax_reference(x_rows)
    # 逐行检查 y 是否随 base 排序递增 (即: y[:, 0] < y[:, 1] < ... < y[:, N-1])
    diffs = np.diff(y, axis=-1)  # (8, 31)
    # 因为 base 严格递增, softmax 对单调映射保持保序 (同一行内)
    assert np.all(diffs > 0), f"同一行内输入严格递增时 softmax 输出也应严格递增, " \
                               f"min_diff={float(diffs.min()):.8f}"


if __name__ == "__main__":
    # 允许直接 `python test_softmax.py` 跑, 不用装 pytest 也能跑一遍 smoke。
    for shape in [(128,), (16, 64), (2, 4, 32, 128)]:
        test_row_sum_equals_one(np.random.default_rng(42), shape)
    for shape in [(256,), (8, 32), (2, 2, 16, 256)]:
        test_non_negative(np.random.default_rng(42), shape)
    for dt in (np.float32, np.float64, np.float16):
        test_preserves_dtype(np.random.default_rng(42), dt)
    test_numerical_stability_vs_naive(np.random.default_rng(42))
    test_gradient_direction(np.random.default_rng(42))
    print("python-softmax smoke tests PASSED")
