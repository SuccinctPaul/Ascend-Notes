"""
GELU (TileLang-Ascend) 正确性测试占位。

TileLang 本身是 Python → TVM/TIR → AscendC 的跨平台编译流程, 但真正跑在 NPU 上
依然需要 CANN + NPU 硬件。这里的策略:
  - 本地只有 Python + tilelang 包时: 至少验证能成功 import 并能创建 kernel 对象。
  - 有 NPU/CANN 环境时: 再跑真实数值对比 (此时 TileLang 会完整走完编译→运行流程)。
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

    # 基本 sanity: 能构造出 kernel 对象。
    from gelu_tilelang import gelu_activation, gelu_tilelang, gelu_reference_numpy

    _ = gelu_activation(4096, 1024, dtype="float16")  # 不触发 JIT 运行, 只拿 Python 对象

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(7)
    failed = 0
    cases = [
        (4096,  1024),
        (16384, 2048),
        (1024,  256),
    ]
    for N, BLOCK in cases:
        x = (rng.standard_normal(N) * 3.0).astype(np.float16)
        y = gelu_tilelang(x, BLOCK=BLOCK)
        ref = gelu_reference_numpy(x)
        max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
        ok = max_err < 5e-3
        print(f"[{'PASS' if ok else 'FAIL'}] N={N:<6d} BLOCK={BLOCK:<4d} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
