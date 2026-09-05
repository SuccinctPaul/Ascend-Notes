"""
Quant (INT8 对称量化) correctness test —— 纯 Python/NumPy 参考实现。

校验维度:
  1) 值域: q ∈ [-127, 127]
  2) 往返误差: max|x - dequant(quant(x))| ≤ 每行 scale (且典型 ≈ scale/2)
  3) amax 元素量化后 = ±127
  4) dtype: q=int8, scale=fp32, dequant 输出 fp16
  5) 全零行 (scale 防除零) 不产生 nan
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except Exception:  # pragma: no cover - 允许无 pytest 直接跑 __main__ smoke
    class _FakeMark:
        def __getattr__(self, name):
            def _deco(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]
                def wrap(f):
                    return f
                return wrap
            return _deco
    class _FakePytest:
        mark = _FakeMark()
        @staticmethod
        def fixture(*args, **kwargs):
            def wrap(f):
                return f
            return wrap if not (len(args) == 1 and callable(args[0]) and not kwargs) else args[0]
    pytest = _FakePytest()

from quant import (quant_int8_reference, dequant_int8_reference,
                   roundtrip_max_error, QMAX)


def _assert_props(x: np.ndarray) -> None:
    q, scale = quant_int8_reference(x)
    assert q.dtype == np.int8 and scale.dtype == np.float32
    assert q.min() >= -QMAX and q.max() <= QMAX, "q 必须落在 [-127, 127]"
    # 往返误差 ≤ 每行 scale
    xh = dequant_int8_reference(q, scale)
    err = np.abs(xh.astype(np.float32) - x.astype(np.float32))
    per_row_err = err.reshape(-1, x.shape[-1]).max(axis=-1)
    assert np.all(per_row_err <= scale + 1e-6), "往返误差不得超过每行 scale"
    # amax 元素 = ±127
    xf = x.astype(np.float32).reshape(-1, x.shape[-1])
    q2 = q.reshape(-1, x.shape[-1])
    for r in range(xf.shape[0]):
        amax_idx = int(np.argmax(np.abs(xf[r])))
        assert abs(int(q2[r, amax_idx])) == QMAX


def test_basic_shapes():
    rng = np.random.default_rng(42)
    for shape in [(16, 128), (64, 512), (8, 4096)]:
        _assert_props((rng.standard_normal(shape) * 2.0).astype(np.float16))


def test_extreme_values():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((32, 256)) * 100.0).astype(np.float16)
    _assert_props(x)


def test_zero_row_no_nan():
    x = np.zeros((4, 128), dtype=np.float16)
    q, scale = quant_int8_reference(x)
    assert np.isfinite(scale).all(), "全零行 scale 应为有限值 (防除零)"
    xh = dequant_int8_reference(q, scale)
    assert np.isfinite(xh).all()


def test_dtype_preserved():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((8, 64)) * 2.0).astype(np.float16)
    xh = dequant_int8_reference(*quant_int8_reference(x))
    assert xh.dtype == np.float16


def test_roundtrip_error_bound():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((16, 512)) * 2.0).astype(np.float16)
    err = roundtrip_max_error(x)
    # 典型往返误差 ≈ scale/2; 这里宽松断言 ≤ scale
    _, scale = quant_int8_reference(x)
    assert err <= float(np.max(scale))


if __name__ == "__main__":
    test_basic_shapes()
    test_extreme_values()
    test_zero_row_no_nan()
    test_dtype_preserved()
    test_roundtrip_error_bound()
    print("python-quant smoke tests PASSED")
