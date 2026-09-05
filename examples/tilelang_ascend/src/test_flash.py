"""
FlashAttention (TileLang-Ascend) 正确性测试。

对齐 test_gqa.py (tilelang) 风格: 教学版逐行串行, 用小档位;
无 tilelang / 无 NPU 时 SKIP。
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

    from flash_tilelang import flash_attention, flash_attention_tilelang, \
        attention_reference_numpy

    _ = flash_attention(2, 8, 64, 64)   # 只拿 Python 对象, 不触发 JIT 运行

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(7)
    failed = 0
    # (name, H, L, S, D) — 教学版逐行串行, 小档位
    cases = [
        ("flash-1x8x64x64",   1, 8, 64, 64),
        ("flash-2x8x64x64",   2, 8, 64, 64),
        ("flash-2x16x100x64", 2, 16, 100, 64),   # S 非 2 幂
    ]
    for name, H, L, S, D in cases:
        q = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
        k = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
        v = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
        out = flash_attention_tilelang(q, k, v)
        ref = attention_reference_numpy(q, k, v)
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
        ok = max_err < 2e-2
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<18s} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
