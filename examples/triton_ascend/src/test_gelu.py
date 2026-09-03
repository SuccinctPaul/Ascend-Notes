"""
GELU (Triton-Ascend) 正确性测试占位。

- 有 NPU + triton-ascend 时, 跑真实 kernel, 与 numpy GELU 参考对齐。
- 没有 NPU 时, 打印 SKIP 并退出 0 (防止 CI 在 x86/mac 上无意义失败)。
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
        print("SKIP: NPU not available (test_gelu_triton needs torch.npu + triton-ascend)")
        return 0

    import torch
    from gelu_triton import gelu_triton, gelu_reference_numpy

    rng = np.random.default_rng(0)

    # 三种典型形状 / 两种 dtype
    cases = [
        ("vec-1024",    (1024,),                    torch.float16),
        ("mat-32x1024", (32, 1024),                 torch.float16),
        ("tensor-4d",   (2, 8, 16, 256),            torch.float32),
        ("odd-n",       (12345,),                   torch.float16),   # 不能被 block 整除
    ]

    failed = 0
    for name, shape, dtype in cases:
        x_np = (rng.standard_normal(shape) * 3.0).astype(
            np.float16 if dtype == torch.float16 else np.float32
        )
        ref = gelu_reference_numpy(x_np)

        x_dev = torch.from_numpy(x_np.copy()).npu()
        y_dev = gelu_triton(x_dev)
        y_np = y_dev.cpu().numpy()

        # Triton 当前实现对 fp32 输入也走 fp16 kernel 内部精度 (flat.to(fp16) + 回 cast),
        # 因此实际误差是 fp16 roundtrip 量级 (~10^-3 ~ 10^-2). 两边统一用 fp16 的
        # tolerance 做比较; 将来单独写 fp32 kernel 再收紧.
        atol = 5e-3
        rtol = 5e-3
        diff = np.abs(y_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())

        ok = max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<12s} dtype={dtype} shape={shape!s:<18s} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
