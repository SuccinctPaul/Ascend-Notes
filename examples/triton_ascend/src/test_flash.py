"""
FlashAttention (Triton-Ascend) 正确性测试。

用例覆盖: 常规档 / L 非 BLOCK_M 倍数 / S 非 BLOCK_N 倍数 / D=128 /
多头多块 grid / 大幅值数值稳定。
校验: 与 numpy 标准注意力参考的 allclose (atol=rtol=5e-3, 数学等价)。
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
        print("SKIP: NPU not available (test_flash needs torch.npu + triton-ascend)")
        return 0

    import torch
    from flash_triton import flash_attention_triton, attention_reference_numpy

    rng = np.random.default_rng(0xC0FFEE)

    # (name, H, L, S, D)
    cases = [
        ("flash-2x128x256x64",  2, 128, 256, 64),
        ("flash-4x256x256x128", 4, 256, 256, 128),
        ("l-nonal-2x100x128x64", 2, 100, 128, 64),   # L 非 BLOCK_M 倍数
        ("s-nonal-2x128x200x64", 2, 128, 200, 64),   # S 非 BLOCK_N 倍数
        ("single-1x64x64x64",   1, 64, 64, 64),      # 单块
        ("extreme-2x128x256x64", 2, 128, 256, 64),   # 大幅值数值稳定
    ]

    atol = 5e-3
    rtol = 5e-3
    failed = 0
    for name, H, L, S, D in cases:
        amt = 8.0 if name.startswith("extreme") else 1.5
        q_np = (rng.standard_normal((H, L, D)) * amt).astype(np.float16)
        k_np = (rng.standard_normal((H, S, D)) * amt).astype(np.float16)
        v_np = (rng.standard_normal((H, S, D)) * amt).astype(np.float16)

        ref = attention_reference_numpy(q_np, k_np, v_np)
        out_dev = flash_attention_triton(torch.from_numpy(q_np).npu(),
                                         torch.from_numpy(np.ascontiguousarray(k_np)).npu(),
                                         torch.from_numpy(np.ascontiguousarray(v_np)).npu())
        out_np = out_dev.cpu().numpy()

        diff = np.abs(out_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())
        finite = bool(np.isfinite(out_np).all())
        ok = finite and max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<22s} max_err={max_err:.6e} finite={finite}")
        if not ok:
            failed += 1

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
