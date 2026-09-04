"""
RoPE (TileLang-Ascend) 正确性测试。

对齐 test_softmax.py (tilelang) 风格:
  - 无 tilelang 包时 SKIP;
  - 无 NPU 时只验证 kernel 对象可构造;
  - 有 NPU 时跑多组 D 的数值对比 (atol=1e-2, fp16 三角表 + 乘加放宽)。
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

    from rope_tilelang import rope_2d, rope_tilelang, rope_reference_numpy

    _ = rope_2d(4, 256, dtype="float16")  # 只拿 Python 对象

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(11)
    failed = 0
    # 不同 head/hidden 维: D 必须 = 2*HALF
    cases = [64, 128, 512, 2048]
    for D in cases:
        x = (rng.standard_normal(D) * 2.0).astype(np.float16)
        pos = int(rng.integers(0, 4096))
        y = rope_tilelang(x, np.array([pos]))
        ref = rope_reference_numpy(x, np.array([pos]))
        max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
        ok = max_err < 1e-2
        print(f"[{'PASS' if ok else 'FAIL'}] D={D:<6d} pos={pos:<6d} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    # 2D 输入 (M, D) + 默认递增位置 + 大位置
    M, D = 8, 128
    x2 = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
    pos2 = rng.integers(0, 8192, size=M)
    y2 = rope_tilelang(x2, pos2)
    ref2 = rope_reference_numpy(x2, pos2)
    max_err2 = float(np.max(np.abs(y2.astype(np.float32) - ref2.astype(np.float32))))
    ok2 = max_err2 < 1e-2
    print(f"[{'PASS' if ok2 else 'FAIL'}] 2D ({M}x{D}) positions∈[0,8192) max_err={max_err2:.6e}")
    if not ok2:
        failed += 1

    # 保范数 sanity: 每对范数旋转前后不变
    x3 = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
    y3 = rope_tilelang(x3, np.arange(M))
    n_in = np.sqrt(x3.astype(np.float32)[:, 0::2] ** 2 + x3.astype(np.float32)[:, 1::2] ** 2)
    n_out = np.sqrt(y3.astype(np.float32)[:, 0::2] ** 2 + y3.astype(np.float32)[:, 1::2] ** 2)
    norm_drift = float(np.max(np.abs(n_in - n_out)))
    ok3 = norm_drift < 5e-2
    print(f"[{'PASS' if ok3 else 'FAIL'}] norm-preservation drift={norm_drift:.6e}")
    if not ok3:
        failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
