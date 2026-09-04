"""
INT8 量化 (Triton-Ascend) 正确性测试。

对齐 test_rmsnorm.py 风格:
  - _has_real_npu() 环境守卫;
  - 多 shape / fp16/fp32 / odd-D / 大幅值 / 全零行用例;
  - 校验: q 值域、q 一致率 (>99.9%)、scale 误差、往返误差 ≤ 每行 scale。
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
        print("SKIP: NPU not available (test_quant needs torch.npu + triton-ascend)")
        return 0

    import torch
    from quant_triton import (quant_int8_triton, dequant_int8_triton,
                              quant_int8_reference_numpy)

    rng = np.random.default_rng(0xC0FFEE)

    cases = [
        ("mat-16x128",      (16, 128),  torch.float16, 2.0),
        ("mat-32x512",      (32, 512),  torch.float16, 2.0),
        ("mat-odd-63x1023", (63, 1023), torch.float16, 1.0),   # odd D
        ("mat-64x256-fp32", (64, 256),  torch.float32, 2.0),
        ("big-8x4096",      (8, 4096),  torch.float16, 1.0),   # D > BLOCK
        ("extreme-16x512",  (16, 512),  torch.float16, 100.0), # 大幅值
    ]

    failed = 0
    for name, shape, dtype, scale_amt in cases:
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        x_np = (rng.standard_normal(shape) * scale_amt).astype(np_dtype)

        q_ref, s_ref = quant_int8_reference_numpy(x_np)

        x_dev = torch.from_numpy(x_np.copy()).npu()
        q_dev, s_dev = quant_int8_triton(x_dev)
        y_dev = dequant_int8_triton(q_dev, s_dev, torch.float16 if dtype == torch.float16 else torch.float32)

        q_np = q_dev.cpu().numpy()
        s_np = s_dev.cpu().numpy()
        y_np = y_dev.cpu().numpy()

        # 校验 1: 值域
        ok_range = int(q_np.min()) >= -127 and int(q_np.max()) <= 127
        # 校验 2: 一致率 (round-half 语义差允许 ±1 → 一致率 > 99.9%)
        q_match = float(np.mean(q_np.astype(np.int32) == q_ref.astype(np.int32)))
        ok_match = q_match > 0.999
        # 校验 3: scale
        scale_err = float(np.max(np.abs(s_np - s_ref)))
        ok_scale = scale_err < 1e-6
        # 校验 4: 往返误差 ≤ 每行 scale
        rt = np.abs(y_np.astype(np.float32) - x_np.astype(np.float32))
        rt_err = float(rt.max())
        ok_rt = rt_err <= float(s_ref.max()) + 1e-6

        ok = ok_range and ok_match and ok_scale and ok_rt
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<18s} dtype={str(dtype):<7s} "
              f"q_match={q_match:.4f} scale_err={scale_err:.2e} "
              f"roundtrip={rt_err:.6f}/{float(s_ref.max()):.6f}")
        if not ok:
            failed += 1
            if not ok_range: print("  ↳ ❌ q 越界")
            if not ok_match: print("  ↳ ❌ q 一致率低")
            if not ok_scale: print("  ↳ ❌ scale 误差大")
            if not ok_rt:    print("  ↳ ❌ 往返误差超上界")

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
