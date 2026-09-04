"""
RMSNorm benchmark —— 在 NPU 上跑 Triton-Ascend rmsnorm_triton, 多种 (M, D), 多轮.

用法:
  python3 bench_rmsnorm_triton.py [--device 0] [--rows 1024,4096,16384] \
      [--dims 512,1024,4096] [--repeats 20] [--block-size 1024]

输出: CSV 到 stdout + JSON 摘要到 stderr.
Roofline 视角: RMSNorm 每元素 ~6 FLOPs (square/mul/mul + 归约摊销),
访存 = 读 x + 读 gamma(摊销) + 写 y ≈ 4 B/元素 (fp16), 典型的带宽受限算子.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from rmsnorm_triton import rmsnorm_triton, rmsnorm_reference_numpy

HBM_TBPS = 1.6      # 910B2 单 chip HBM2e 标称带宽 (TB/s)


def bench_one(x_npu, g_npu, warmup: int, repeats: int) -> float:
    """返回最快一轮的毫秒 (下界, 避免系统抖动拉高)."""
    import torch
    for _ in range(warmup):
        _ = rmsnorm_triton(x_npu, g_npu)
    torch.npu.synchronize()

    best_ms = float("inf")
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        _ = rmsnorm_triton(x_npu, g_npu)
        torch.npu.synchronize()
        t1 = time.perf_counter_ns()
        ms = (t1 - t0) * 1e-6
        if ms < best_ms:
            best_ms = ms
    return best_ms


def main() -> int:
    ap = argparse.ArgumentParser(description="Triton-Ascend RMSNorm 微基准.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rows", type=str, default="1024,4096,16384")
    ap.add_argument("--dims", type=str, default="512,1024,4096")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.npu.set_device(args.device)

    rows = [int(s) for s in args.rows.split(",") if s]
    dims = [int(s) for s in args.dims.split(",") if s]
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    bytes_elem = 2 if args.dtype == "fp16" else 4

    rng = np.random.default_rng(0xC0FFEE)
    summary = []
    print("M,D,bytes,ms_best,GBps,GFLOPS_elem,correctness_max_abs")

    for M in rows:
        for D in dims:
            x_np = (rng.standard_normal((M, D)) * 2.0).astype(
                np.float16 if args.dtype == "fp16" else np.float32
            )
            gamma_np = rng.uniform(0.5, 2.0, D).astype(
                np.float16 if args.dtype == "fp16" else np.float32
            )
            x_dev = torch.from_numpy(x_np.copy()).npu()
            g_dev = torch.from_numpy(gamma_np.copy()).npu()

            # correctness (每规模先校验一次)
            y_dev = rmsnorm_triton(x_dev, g_dev, BLOCK_SIZE=args.block_size)
            y_ref = rmsnorm_reference_numpy(x_np, gamma_np)
            y_cpu = y_dev.cpu().numpy()
            max_abs = float(np.max(np.abs(
                y_cpu.astype(np.float32) - y_ref.astype(np.float32)
            )))
            assert max_abs < 5e-2, f"FAIL correctness M={M} D={D} max_abs={max_abs}"

            ms = bench_one(x_dev, g_dev, args.warmup, args.repeats)
            # 访存: 读 x + 写 y + 读 gamma (摊销, 忽略) → 2×bytes/元素
            bytes_total = M * D * bytes_elem * 2
            gbps = bytes_total / (ms * 1e-3) / 1e9
            # FLOPs: 平方/乘/加 ~6 FLOPs/元素 (归约摊销)
            flops = M * D * 6
            gflops = flops / (ms * 1e-3) / 1e9
            print(f"{M},{D},{bytes_total},{ms:.4f},{gbps:.2f},{gflops:.1f},{max_abs:.2e}")
            sys.stdout.flush()
            summary.append(dict(M=M, D=D, bytes_total=bytes_total, ms_best=round(ms, 4),
                                GBps=round(gbps, 3), GFLOPS=round(gflops, 2),
                                max_abs=max_abs))

    obj = dict(benchmark="rmsnorm_triton_ascend",
               SoC="910B2", device_id=args.device,
               dtype=args.dtype, block_size=args.block_size,
               HBM_TBPS_quoted=HBM_TBPS,
               rows=summary)
    print("\n" + json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
