"""
INT8 量化 (TileLang-Ascend) 正确性测试。

对齐 test_rmsnorm.py (tilelang) 风格:
  - 无 tilelang 包时 SKIP;
  - 无 NPU 时只验证 kernel 对象可构造;
  - 有 NPU 时跑多组 (M, D) 的数值对比。
"""

from __future__ import annotations

import sys
import numpy as np


def _import_tilelang():
    try:
        import tilelang  # noqa: F401
        import tilelang.language as T  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover
        print(f"SKIP: tilelang not installed: {e}")
        return False


def _has_real_npu():
    try:
        import torch
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _main():
    if not _import_tilelang():
        return 0

    from quant_tilelang import quant_2d, dequant_2d, quant_int8_tilelang, \
        dequant_int8_tilelang, quant_int8_reference_numpy

    _ = quant_2d(4, 256)    # 只拿 Python 对象, 不触发 JIT 运行
    _ = dequant_2d(4, 256, dtype="float16")

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(11)
    failed = 0
    cases = [
        (8, 128, 2.0),
        (16, 512, 2.0),
        (16, 4096, 1.0),
        (32, 256, 100.0),   # 大幅值
    ]
    for M, D, amt in cases:
        x = (rng.standard_normal((M, D)) * amt).astype(np.float16)
        q, scale = quant_int8_tilelang(x)
        q_ref, scale_ref = quant_int8_reference_numpy(x)
        q_match = float(np.mean(q.astype(np.int32) == q_ref.astype(np.int32)))
        scale_err = float(np.max(np.abs(scale - scale_ref)))
        y = dequant_int8_tilelang(q, scale)
        rt = float(np.max(np.abs(y.astype(np.float32) - x.astype(np.float32))))
        # q 允许 ±1 LSB (fp16 中转舍入); 硬标准是往返误差 ≤ max_scale
        ok = (q.min() >= -127 and q.max() <= 127 and q_match > 0.98
              and scale_err < 1e-6 and rt <= float(scale_ref.max()) + 1e-6)
        print(f"[{'PASS' if ok else 'FAIL'}] M={M:<4d} D={D:<6d} amt={amt:<6} "
              f"q_match={q_match:.4f} scale_err={scale_err:.2e} roundtrip={rt:.6f}")
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
