"""
Softmax (Triton-Ascend) 正确性测试。

对齐 test_gelu.py 风格:
  - _has_real_npu() 环境守卫; 无 NPU 时打印 SKIP 并退出 0.
  - 多 shape / 多 dtype / odd-D (padding path) / 大 D (>BLOCK_SIZE, pad path) 用例.
  - 三个校验维度:
      1) 与 numpy reference 的 allclose (atol/rtol=5e-3)
      2) 每行和 ≈ 1.0
      3) 所有元素非负
  - 每个 case 打印 [PASS|FAIL] + max_err / sum_err / min_y
  - 汇总返回 exit(0) 或 exit(1)
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
        print("SKIP: NPU not available (test_softmax needs torch.npu + triton-ascend)")
        return 0

    import torch
    from softmax_triton import softmax_triton, softmax_reference_numpy

    rng = np.random.default_rng(0xC0FFEE)

    # (name, shape, dtype) cases
    # 覆盖: 1D / 2D / 4D, fp16/fp32, odd-D (padding path), 大 D (>BLOCK_SIZE=1024)
    cases = [
        ("vec-128",         (128,),                    torch.float16),
        ("vec-odd-12345",   (12345,),                  torch.float16),  # 1D, D 非对齐
        ("mat-32x128",      (32, 128),                 torch.float16),  # 标准 2D fp16
        ("mat-16x2048",     (16, 2048),                torch.float16),  # D > 1024 pad
        ("mat-64x512-fp32", (64, 512),                 torch.float32),  # 2D fp32
        ("tensor-4d",       (2, 8, 16, 256),           torch.float16),  # 4D, last axis
        ("tensor-4d-odd",   (2, 4, 8, 123),            torch.float16),  # 4D, odd D
        ("big-row-128x4096",(128, 4096),               torch.float16),  # D >> BLOCK_SIZE
        ("batch-1024x768",  (1024, 768),               torch.float16),  # 多行小 D
    ]

    atol = 5e-3
    rtol = 5e-3

    failed = 0
    for name, shape, dtype in cases:
        # 1) 生成 host numpy 输入 (和 reference 同 dtype)
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        x_np = (rng.standard_normal(shape) * 3.0).astype(np_dtype)

        # 2) numpy reference (fp32 中间)
        ref = softmax_reference_numpy(x_np, axis=-1)

        # 3) kernel 跑
        x_dev = torch.from_numpy(x_np.copy()).npu()
        y_dev = softmax_triton(x_dev)
        y_np = y_dev.cpu().numpy()

        # ---------- 校验 1: allclose 误差 ----------
        diff = np.abs(y_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())
        ok_err = max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)

        # ---------- 校验 2: 每行 sum ≈ 1 ----------
        # flatten leading dims → (M, D)
        D = shape[-1]
        y_flat = y_np.astype(np.float32).reshape(-1, D)
        row_sums = y_flat.sum(axis=-1)
        sum_err = float(np.max(np.abs(row_sums - 1.0)))
        ok_sum = sum_err <= 5e-3

        # ---------- 校验 3: 非负 ----------
        min_y = float(y_flat.min())
        ok_nonneg = min_y >= -1e-6   # 允许极小 fp16 下溢误差

        ok = ok_err and ok_sum and ok_nonneg
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name:<18s} dtype={str(dtype):<7s} "
            f"shape={str(shape):<18s} "
            f"max_err={max_err:.6e} "
            f"sum_err={sum_err:.6e} "
            f"min_y={min_y:.6e}"
        )
        if not ok:
            failed += 1
            if not ok_err:    print("  ↳ ❌ allclose failed")
            if not ok_sum:    print("  ↳ ❌ row sum != 1")
            if not ok_nonneg: print("  ↳ ❌ negative prob")

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
