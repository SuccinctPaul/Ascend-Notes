"""
RMSNorm (TileLang-Ascend) 正确性测试。

对齐 test_softmax.py (tilelang) 风格:
  - 无 tilelang 包时 SKIP;
  - 无 NPU 时只验证 kernel 对象可构造;
  - 有 NPU 时跑多组 D 的数值对比 (atol=1e-2, fp16 归约累计误差放宽)。
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

    from rmsnorm_tilelang import rmsnorm_2d, rmsnorm_tilelang, rmsnorm_reference_numpy

    _ = rmsnorm_2d(4, 1024, dtype="float16")  # 只拿 Python 对象, 不触发 JIT 运行

    if not _has_real_npu():
        print("SKIP: real NPU not present, skip TileLang numerical run.")
        print("INFO: kernel object constructed OK → TileLang Python API is callable.")
        return 0

    rng = np.random.default_rng(7)
    failed = 0
    # 不同 hidden size: 常见 LLM d_model 档位
    cases = [128, 512, 1024, 4096]
    for D in cases:
        x = (rng.standard_normal(D) * 2.0).astype(np.float16)
        gamma = rng.uniform(0.5, 2.0, D).astype(np.float16)
        y = rmsnorm_tilelang(x, gamma)
        ref = rmsnorm_reference_numpy(x, gamma)
        max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
        # 仓库统一容差 allclose(atol=1e-2, rtol=1e-2): fp16 输出在大 |y| 处
        # 1 ulp 就是 |y|/1024, 固定 1e-2 绝对容差过紧
        tol = 1e-2 + 1e-2 * float(np.abs(ref.astype(np.float32)).max())
        ok = max_err < tol
        print(f"[{'PASS' if ok else 'FAIL'}] D={D:<6d} max_err={max_err:.6e} (tol={tol:.4f})")
        if not ok:
            failed += 1

    # 2D 输入 (M, D): 每行独立
    M, D = 8, 512
    x2 = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
    gamma2 = rng.uniform(0.5, 2.0, D).astype(np.float16)
    y2 = rmsnorm_tilelang(x2, gamma2)
    ref2 = rmsnorm_reference_numpy(x2, gamma2)
    max_err2 = float(np.max(np.abs(y2.astype(np.float32) - ref2.astype(np.float32))))
    tol2 = 1e-2 + 1e-2 * float(np.abs(ref2.astype(np.float32)).max())
    ok2 = max_err2 < tol2
    print(f"[{'PASS' if ok2 else 'FAIL'}] 2D ({M}x{D}) max_err={max_err2:.6e} (tol={tol2:.4f})")
    if not ok2:
        failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
