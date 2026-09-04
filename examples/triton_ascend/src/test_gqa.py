"""
GQA 解码注意力 (Triton-Ascend) 正确性测试。

用例覆盖: GQA 中间态 / MQA 退化 (Hkv=1) / MHA 退化 (Hkv=Hq) / 多 S 档位 /
S 非 BLOCK_S 倍数 / D=64&128。
校验: 与 numpy 参考的 allclose (atol=rtol=5e-3)。
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
        print("SKIP: NPU not available (test_gqa needs torch.npu + triton-ascend)")
        return 0

    import torch
    from gqa_triton import gqa_decode_triton, gqa_decode_reference_numpy

    rng = np.random.default_rng(0xC0FFEE)

    # (name, Hq, Hkv, S, D)
    cases = [
        ("gqa-8x2x256x128",  8, 2, 256, 128),
        ("gqa-8x4x512x64",   8, 4, 512, 64),
        ("mqa-8x1x128x64",   8, 1, 128, 64),    # MQA 退化
        ("mha-4x4x200x128",  4, 4, 200, 128),   # MHA 退化 + S 非 BLOCK_S 倍数
        ("gqa-8x2x1024x64",  8, 2, 1024, 64),   # 长 cache
        ("llama-gqa-8x8x300x128", 8, 8, 300, 128),  # Hq=Hkv=8 (常见 7B 配置按 MHA 用)
    ]

    atol = 5e-3
    rtol = 5e-3
    failed = 0
    for name, Hq, Hkv, S, D in cases:
        q_np = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
        k_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        v_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)

        ref = gqa_decode_reference_numpy(q_np, k_np, v_np)
        out_dev = gqa_decode_triton(torch.from_numpy(q_np).npu(),
                                    torch.from_numpy(np.ascontiguousarray(k_np)).npu(),
                                    torch.from_numpy(np.ascontiguousarray(v_np)).npu())
        out_np = out_dev.cpu().numpy()

        diff = np.abs(out_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())
        ok = max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<22s} max_err={max_err:.6e}")
        if not ok:
            failed += 1

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
