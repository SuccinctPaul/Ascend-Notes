"""
RoPE correctness test —— 纯 Python/NumPy 参考实现。

验证 rope_reference 的数学性质 + 与 PyTorch HF transformers 风格实现对齐:
  1) 旋转保范数: 每对 (x[2a], x[2a+1]) 的欧氏范数旋转前后不变
  2) 相对位置性质: <R(m)q, R(n)k> == <q, R(n-m)k>
  3) vs torch 复数乘参考实现 (float32 / float16)
  4) dtype 保持
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

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from rope import (rope_reference, apply_rope_numpy,
                  precompute_rope_tables, precompute_rope_inv_freq)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _torch_rope_ref(x_np: np.ndarray, positions_np: np.ndarray,
                    base: float = 10000.0) -> np.ndarray:
    """torch 复数乘参考: view_as_complex 实现交错配对旋转 (fp32)."""
    d = x_np.shape[-1]
    inv_freq = torch.from_numpy(precompute_rope_inv_freq(d, base))
    pos = torch.from_numpy(positions_np.astype(np.float64))
    angles = pos[:, None] * inv_freq[None, :]               # (T, d/2)
    cc = torch.polar(torch.ones_like(angles), angles)        # e^{iθ}
    xt = torch.from_numpy(x_np.astype(np.float32)).reshape(-1, d)
    T = xt.shape[0]
    xc = torch.view_as_complex(xt.float().reshape(T, d // 2, 2))
    yr = torch.view_as_real(xc * cc.unsqueeze(0)).reshape(T, d)
    return yr.numpy().reshape(x_np.shape)


# ---------- 1. 旋转保范数 ----------

@pytest.mark.parametrize("d", [64, 128, 512])
def test_norm_preserved(rng, d):
    x = rng.standard_normal((8, d)).astype(np.float32)
    y = apply_rope_numpy(x, np.arange(8))
    n_in = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
    n_out = np.sqrt(y[..., 0::2] ** 2 + y[..., 1::2] ** 2)
    np.testing.assert_allclose(n_in, n_out, atol=1e-5)


# ---------- 2. 相对位置性质 ----------

def test_relative_position_property(rng):
    d = 128
    q = rng.standard_normal(d).astype(np.float32)
    k = rng.standard_normal(d).astype(np.float32)
    m, n = 3, 17
    rq = apply_rope_numpy(q[None], np.array([m]))[0]
    rk = apply_rope_numpy(k[None], np.array([n]))[0]
    rk_rel = apply_rope_numpy(k[None], np.array([n - m]))[0]
    lhs = float(rq @ rk)
    rhs = float(q @ rk_rel)
    assert np.isclose(lhs, rhs, atol=1e-4), f"<R(m)q,R(n)k>={lhs} != <q,R(n-m)k>={rhs}"


# ---------- 3. fp32 vs torch 复数乘 ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("d", [64, 128, 256])
def test_fp32_matches_torch_complex(rng, d):
    x = rng.standard_normal((16, d)).astype(np.float32)
    pos = rng.integers(0, 512, size=16)
    y_ref = apply_rope_numpy(x, pos)
    y_th = _torch_rope_ref(x, pos)
    np.testing.assert_allclose(y_ref, y_th, atol=1e-5, rtol=1e-5)


# ---------- 4. fp16 查表版 vs fp32 计算 (容差放宽) ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fp16_table_lookup(rng):
    d, T = 128, 32
    x = (rng.standard_normal((T, d)) * 2.0).astype(np.float16)
    pos = np.arange(T)
    cos, sin = precompute_rope_tables(pos, d)
    y_ref16 = rope_reference(x, cos.astype(np.float16), sin.astype(np.float16))
    y_th = _torch_rope_ref(x.astype(np.float32), pos).astype(np.float16)
    np.testing.assert_allclose(y_ref16.astype(np.float32),
                               y_th.astype(np.float32), atol=1e-2, rtol=1e-2)


# ---------- 5. dtype 保持 ----------

@pytest.mark.parametrize("dt", [np.float32, np.float64, np.float16])
def test_preserves_dtype(rng, dt):
    d = 64
    x = (rng.standard_normal((4, d)) * 2.0).astype(dt)
    cos, sin = precompute_rope_tables(np.arange(4), d)
    y = rope_reference(x, cos, sin)
    assert y.dtype == dt, f"输入 {dt}, 输出应为 {dt}, 实际 {y.dtype}"


if __name__ == "__main__":
    for d in (64, 128, 512):
        test_norm_preserved(np.random.default_rng(42), d)
    test_relative_position_property(np.random.default_rng(42))
    if _HAS_TORCH:
        for d in (64, 128):
            test_fp32_matches_torch_complex(np.random.default_rng(42), d)
        test_fp16_table_lookup(np.random.default_rng(42))
    for dt in (np.float32, np.float64, np.float16):
        test_preserves_dtype(np.random.default_rng(42), dt)
    print("python-rope smoke tests PASSED")
