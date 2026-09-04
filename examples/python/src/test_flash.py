"""
FlashAttention correctness test —— 纯 Python/NumPy 参考实现。

核心校验: **flash online softmax 与标准注意力数学等价** (逐块增量 == 全量 softmax),
另附数值稳定性 (大幅值输入) 与 dtype 保持。
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except Exception:  # pragma: no cover
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

from flash import attention_reference, flash_online_reference


@pytest.mark.parametrize("BLOCK_N", [16, 64, 128])
def test_online_equals_standard(BLOCK_N):
    rng = np.random.default_rng(42)
    H, L, S, D = 2, 32, 64, 64
    q = rng.standard_normal((H, L, D)).astype(np.float16)
    k = rng.standard_normal((H, S, D)).astype(np.float16)
    v = rng.standard_normal((H, S, D)).astype(np.float16)
    ref = attention_reference(q, k, v)
    online = flash_online_reference(q, k, v, BLOCK_N=BLOCK_N)
    err = float(np.max(np.abs(ref.astype(np.float32) - online.astype(np.float32))))
    assert err < 5e-3, f"online vs standard err={err}"


def test_numerical_stability():
    rng = np.random.default_rng(7)
    q = (rng.standard_normal((2, 32, 64)) * 8.0).astype(np.float16)
    k = (rng.standard_normal((2, 64, 64)) * 8.0).astype(np.float16)
    v = (rng.standard_normal((2, 64, 64)) * 8.0).astype(np.float16)
    out = attention_reference(q, k, v)
    assert np.isfinite(out).all(), "大幅值输入 (score ~ ±50) 输出应为有限值"


def test_dtype_preserved():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((1, 16, 64)).astype(np.float16)
    k = rng.standard_normal((1, 32, 64)).astype(np.float16)
    v = rng.standard_normal((1, 32, 64)).astype(np.float16)
    out = attention_reference(q, k, v)
    assert out.dtype == np.float16 and out.shape == q.shape


if __name__ == "__main__":
    for bn in (16, 64, 128):
        test_online_equals_standard(bn)
    test_numerical_stability()
    test_dtype_preserved()
    print("python-flash smoke tests PASSED")
