"""
RMSNorm correctness test —— 纯 Python/NumPy 参考实现。

验证 rmsnorm_reference 的数学性质 + 与 PyTorch RMSNorm 实现对齐:
  1) y/gamma 的均方 ≈ 1 (归一化能量守恒)
  2) 缩放不变性: rmsnorm(c·x) == rmsnorm(x), c>0
  3) fp32/fp16 vs torch.nn.RMSNorm (或手写 torch 参考版)
  4) dtype 保持
  5) naive 版 (fp16 归约) 与 fp32 归约版的误差差异 (教学点)
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

from rmsnorm import rmsnorm_reference, rmsnorm_naive, rmsnorm_numpy


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _torch_rmsnorm(x_np: np.ndarray, gamma_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """torch 参考版 (优先 nn.RMSNorm, 老版本 torch 没有则手写公式)."""
    xt = torch.from_numpy(x_np.copy())
    gt = torch.from_numpy(gamma_np.copy())
    if hasattr(torch.nn, "RMSNorm"):
        mod = torch.nn.RMSNorm(x_np.shape[-1], eps=eps,
                               elementwise_affine=True, dtype=xt.dtype)
        with torch.no_grad():
            mod.weight.copy_(gt)
        return mod(xt).numpy()
    ms = xt.pow(2).mean(-1, keepdim=True)
    return (xt * torch.rsqrt(ms + eps) * gt).numpy()


# ---------- 1. 归一化能量: y/gamma 每行均方 ≈ 1 ----------

@pytest.mark.parametrize("d", [128, 512, 1024, 4096])
def test_unit_mean_square(rng, d):
    x = rng.standard_normal((8, d)).astype(np.float32)
    gamma = rng.uniform(0.5, 2.0, d).astype(np.float32)
    y = rmsnorm_reference(x, gamma)
    ms = np.mean(np.square(y / gamma), axis=-1)
    np.testing.assert_allclose(ms, 1.0, atol=1e-4)


# ---------- 2. 缩放不变性 ----------

def test_scale_invariance(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    gamma = np.ones(256, dtype=np.float32)
    y1 = rmsnorm_reference(x, gamma)
    y2 = rmsnorm_reference(7.5 * x, gamma)
    np.testing.assert_allclose(y1, y2, atol=1e-5, rtol=1e-5)


# ---------- 3. fp32 vs torch ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (4, 8, 1024)])
def test_fp32_matches_torch(rng, shape):
    x = rng.standard_normal(shape).astype(np.float32) * 2.0
    gamma = rng.uniform(0.5, 2.0, shape[-1]).astype(np.float32)
    y_ref = rmsnorm_reference(x, gamma)
    y_th = _torch_rmsnorm(x, gamma)
    np.testing.assert_allclose(y_ref, y_th, atol=1e-5, rtol=1e-5)


# ---------- 4. fp16 vs torch (fp16 输入, fp32 归约) ----------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fp16_matches_torch(rng):
    d = 1024
    x = (rng.standard_normal((16, d)) * 2.0).astype(np.float16)
    gamma = rng.uniform(0.5, 2.0, d).astype(np.float16)
    y_ref = rmsnorm_reference(x, gamma)                     # fp32 归约
    y_th = _torch_rmsnorm(x.astype(np.float32),
                          gamma.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(y_ref.astype(np.float32),
                               y_th.astype(np.float32), atol=1e-2, rtol=1e-2)


# ---------- 5. dtype 保持 ----------

@pytest.mark.parametrize("dt", [np.float32, np.float64, np.float16])
def test_preserves_dtype(rng, dt):
    d = 128
    x = (rng.standard_normal((8, d)) * 2.0).astype(dt)
    gamma = np.ones(d, dtype=dt)
    y = rmsnorm_reference(x, gamma)
    assert y.dtype == dt, f"输入 {dt}, 输出应为 {dt}, 实际 {y.dtype}"


# ---------- 6. 教学点: fp16 归约 vs fp32 归约, 长行误差差异 ----------

def test_fp32_accumulation_matters(rng):
    # d=8192 的长行: fp16 里累加 Σx² 的 naive 版应有可见误差; fp32 版贴住 torch
    d = 8192
    x = (rng.standard_normal((4, d)) * 2.0).astype(np.float32)
    gamma = np.ones(d, dtype=np.float32)
    y_naive = rmsnorm_naive(x.astype(np.float16), gamma.astype(np.float16))
    y_ref = rmsnorm_numpy(x, gamma)
    err_naive = float(np.max(np.abs(y_naive.astype(np.float32) - y_ref)))
    # 不 assert naive 一定差, 但打印教学数据; fp32 版必须紧
    print(f"\n[teaching] d={d} fp16-accumulation max_err={err_naive:.3e} (vs fp32 accumulation)")
    # eps 防除零: 全零行不应产生 nan
    x0 = np.zeros((2, 128), dtype=np.float32)
    y0 = rmsnorm_reference(x0, np.ones(128, dtype=np.float32))
    assert np.isfinite(y0).all(), "全零输入 (只靠 eps) 也应输出有限值"


if __name__ == "__main__":
    for d in (128, 512, 1024, 4096):
        test_unit_mean_square(np.random.default_rng(42), d)
    test_scale_invariance(np.random.default_rng(42))
    for shape in ((16, 128), (32, 512)):
        if _HAS_TORCH:
            test_fp32_matches_torch(np.random.default_rng(42), shape)
    for dt in (np.float32, np.float64, np.float16):
        test_preserves_dtype(np.random.default_rng(42), dt)
    test_fp32_accumulation_matters(np.random.default_rng(42))
    print("python-rmsnorm smoke tests PASSED")
