"""
GQA 解码注意力 (TileLang-Ascend) 正确性测试。

对齐 test_rmsnorm.py (tilelang) 风格:
  - 无 tilelang 包时 SKIP;
  - 无 NPU 时只验证 kernel 对象可构造;
  - 有 NPU 时跑 GQA/MQA/MHA 退化的数值对比。
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

    from gqa_tilelang import gqa_decode, gqa_decode_tilelang, gqa_decode_reference_numpy

    _ = gqa_decode(4, 2, 128, 64)   # 只拿 Python 对象, 不触发 JIT 运行

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(7)
    failed = 0
    # (name, Hq, Hkv, S, D) — 教学版串行实现, S/D 取小档位
    cases = [
        ("gqa-4x2x64x64",   4, 2, 64, 64),
        ("gqa-8x2x128x64",  8, 2, 128, 64),
        ("mqa-4x1x64x128",  4, 1, 64, 128),
        ("mha-4x4x100x64",  4, 4, 100, 64),   # S 非 2 幂
    ]
    for name, Hq, Hkv, S, D in cases:
        q = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
        k = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        v = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        out = gqa_decode_tilelang(q, k, v)
        ref = gqa_decode_reference_numpy(q, k, v)
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
        ok = max_err < 2e-2
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<18s} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
