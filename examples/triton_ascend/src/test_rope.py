"""
RoPE (Triton-Ascend) 正确性测试。

对齐 test_softmax.py 风格:
  - _has_real_npu() 环境守卫; 无 NPU 时打印 SKIP 并退出 0.
  - 多 shape / 多 dtype / odd-D(非 BLOCK 倍数) / 4D / 大 T 用例.
  - 校验维度:
      1) 与 numpy reference 的 allclose (atol=rtol=5e-3)
      2) 旋转保范数: 每对 (y[2a], y[2a+1]) 的范数 == 输入对应对的范数
      3) 随机子集 vs torch 参考跳过 (numpy reference 已覆盖)
  - 每个 case 打印 [PASS|FAIL]; 汇总 exit(0)/exit(1).
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
        print("SKIP: NPU not available (test_rope needs torch.npu + triton-ascend)")
        return 0

    import torch
    from rope_triton import rope_triton, rope_reference_numpy

    rng = np.random.default_rng(0xC0FFEE)

    # (name, shape, dtype) —— 最后一维必须为偶数; 行数 = 前导维乘积
    cases = [
        ("mat-16x128",      (16, 128),        torch.float16),
        ("mat-32x512",      (32, 512),        torch.float16),
        ("mat-odd-31x126",  (31, 126),        torch.float16),  # HALF=63 非 BLOCK 倍数
        ("mat-64x256-fp32", (64, 256),        torch.float32),
        ("tensor-4d",       (2, 4, 8, 128),   torch.float16),  # (B, H, T, D)
        ("qwen-d-4096",     (128, 4096),      torch.float16),  # LLaMA/Qwen 级 head*rot
        ("long-seq-2048",   (2048, 128),      torch.float16),  # 长序列, 大位置角度
    ]

    atol = 5e-3
    rtol = 5e-3

    failed = 0
    for name, shape, dtype in cases:
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        D = shape[-1]
        T = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        x_np = (rng.standard_normal(shape) * 2.0).astype(np_dtype)
        # 大位置混入: 检验远距离角度 (fp16 表 + fp32 角度计算)
        positions = rng.integers(0, 8192, size=T)

        ref = rope_reference_numpy(x_np, positions)

        x_dev = torch.from_numpy(x_np.copy()).npu()
        y_dev = rope_triton(x_dev, positions)
        y_np = y_dev.cpu().numpy()

        # ---------- 校验 1: allclose ----------
        diff = np.abs(y_np.astype(np.float32) - ref.astype(np.float32))
        max_err = float(diff.max())
        ok_err = max_err <= atol + rtol * float(np.abs(ref.astype(np.float32)).max() + 1e-6)

        # ---------- 校验 2: 每对范数守恒 ----------
        x32 = x_np.astype(np.float32).reshape(T, D)
        y32 = y_np.astype(np.float32).reshape(T, D)
        n_in  = np.sqrt(x32[:, 0::2] ** 2 + x32[:, 1::2] ** 2)
        n_out = np.sqrt(y32[:, 0::2] ** 2 + y32[:, 1::2] ** 2)
        norm_err = float(np.max(np.abs(n_in - n_out)))
        ok_norm = norm_err <= 5e-2  # fp16 输出下放宽

        ok = ok_err and ok_norm
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name:<18s} dtype={str(dtype):<7s} "
            f"shape={str(shape):<18s} "
            f"max_err={max_err:.6e} "
            f"norm_drift={norm_err:.6e}"
        )
        if not ok:
            failed += 1
            if not ok_err:  print("  ↳ ❌ allclose failed")
            if not ok_norm: print("  ↳ ❌ per-pair norm not preserved")

    if failed == 0:
        print(f"\nAll {len(cases)} cases PASSED")
    else:
        print(f"\n{failed}/{len(cases)} cases FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
