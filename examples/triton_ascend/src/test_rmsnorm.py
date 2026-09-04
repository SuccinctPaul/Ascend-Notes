"""
RMSNorm (Triton-Ascend) 正确性测试。

对齐 test_softmax.py 风格:
  - _has_real_npu() 环境守卫; 无 NPU 时打印 SKIP 并退出 0.
  - 多 shape / 多 dtype / odd-D / 大 D (>BLOCK_SIZE) 用例.
  - 校验维度:
      1) 与 numpy reference 的 allclose (atol=rtol=5e-3)
      2) 归一化能量: y/gamma 每行均方 ≈ 1
  - 每个 case 打印 [PASS|FAIL]; 汇总 exit(0)/exit(1).
"""

from __future__ import annotations

import sys
import numpy as np


def _has_real_npu():
    try:
        import torch  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _main():
    if not _has_real_npu():
        print("SKIP: NPU not available (test_rmsnorm needs torch.npu + triton-ascend)")
        return 0

    import torch
    from rmsnorm_triton import rmsnorm_triton, rmsnorm_reference_numpy

    rng = np.random.default_rng(0xC0FFEE)

    # (name, shape, dtype, D_expected) —— D = shape[-1]
    cases = [
        ("mat-16x128",       (16, 128),         torch.float16),
        ("mat-32x512",       (32, 512),         torch.float16),
        ("mat-16x2048",      (16, 2048),        torch.float16),  # D > BLOCK_SIZE=1024
        ("mat-64x512-fp32",  (64, 512),         torch.float32),
        ("mat-odd-63x1023",  (63, 1023),        torch.float16),  # odd D (mask path)
        ("tensor-4d",        (2, 8, 16, 256),   torch.float16),  # 4D, last axis
        ("llama-4096",       (128, 4096),       torch.float16),  # LLaMA 级 hidden size
        ("batch-1024x768",   (1024, 768),       torch.float16),
    ]

    atol = 5e-3
    rtol = 5e-3

    failed = 0
    for name, shape, dtype in cases:
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        D = shape[-1]
        x_np = (rng.standard_normal(shape) * 2.0).astype(np_dtype)
        gamma_np = rng.uniform(0.5, 2.0, D).astype(np_dtype)

        ref = rmsnorm_reference_numpy(x_np, gamma_np)

        x_dev = torch.from_numpy(x_np.copy()).npu()
        g_dev = torch.from_numpy(gamma_np.copy()).npu()
        y_dev = rmsnorm_triton(x_dev, g_dev)
        y_np = y_dev.cpu().numpy()

        # ---------- 校验 1: allclose ----------
        diff = np.abs(y_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())
        ok_err = max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)

        # ---------- 校验 2: 归一化能量 y/gamma 均方 ≈ 1 ----------
        y_flat = y_np.astype(np.float32).reshape(-1, D)
        g32 = gamma_np.astype(np.float32)
        ms = np.mean(np.square(y_flat / g32), axis=-1)
        ms_err = float(np.max(np.abs(ms - 1.0)))
        ok_ms = ms_err <= 2e-2  # fp16 输出下放宽

        ok = ok_err and ok_ms
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name:<18s} dtype={str(dtype):<7s} "
            f"shape={str(shape):<18s} "
            f"max_err={max_err:.6e} "
            f"ms_err={ms_err:.6e}"
        )
        if not ok:
            failed += 1
            if not ok_err: print("  ↳ ❌ allclose failed")
            if not ok_ms:  print("  ↳ ❌ mean-square(y/gamma) != 1")

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
