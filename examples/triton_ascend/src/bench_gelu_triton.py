"""
GELU benchmark —— 在 NPU 上跑 Triton-Ascend gelu_triton, 多种 N, 多轮, 输出 Roofline 分析所需字段.

用法:
  python3 bench_gelu_triton.py [--device 2] [--sizes 4096,65536,1048576,8388608,67108864,134217728] [--repeats 20] [--block-size 1024]

输出: CSV 到 stdout + JSON 摘要到 stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# 复用同一台机器上已就位的 gelu_triton + gelu_reference_numpy
from gelu_triton import gelu_triton, gelu_reference_numpy


# 910B2 硬件常数 (公开参数, 文档里 roofline 章节已用)
#   HBM 带宽  ~ 1.6 TB/s (四通道 HBM2e, 每 chip 标称带宽)
#   Vector fp16 FLOPS: ~ 280 TFLOPS 量级 (910B 的 Vector 理论峰值, 单 chip)
#   但 GELU element-wise 的有效 FLOPs/元素很少, 基本一定带宽受限.
HBM_TBPS = 1.6      # TB/s, 单 chip 保守值


def bench_one(x_npu, y_npu, block_size: int, warmup: int, repeats: int) -> float:
    """返回最快一轮的毫秒 (下界, 避免系统抖动拉高)."""
    # warmup
    for _ in range(warmup):
        _ = gelu_triton(x_npu, block_size=block_size)
    # 同步一次以保证队列排空
    x_npu.device
    try:
        import torch
        torch.npu.synchronize()
    except Exception:
        pass

    best_ms = float("inf")
    for _ in range(repeats):
        try:
            import torch
            torch.npu.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter_ns()
        _ = gelu_triton(x_npu, block_size=block_size)
        try:
            import torch
            torch.npu.synchronize()
        except Exception:
            pass
        t1 = time.perf_counter_ns()
        ms = (t1 - t0) * 1e-6
        if ms < best_ms:
            best_ms = ms
    return best_ms


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Triton-Ascend GELU element-wise 微基准, 配合 ASCEND_RT_VISIBLE_DEVICES 使用.")
    ap.add_argument("--device", type=int, default=0,
                    help="逻辑 NPU 编号 (ASCEND_RT_VISIBLE_DEVICES 重映射后的编号). "
                         "推荐: ASCEND_RT_VISIBLE_DEVICES=2  python3 bench_gelu_triton.py --device 0")
    ap.add_argument("--sizes", type=str,
                    default="4096,65536,1048576,8388608,33554432,67108864,134217728")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.npu.set_device(args.device)

    sizes = [int(s) for s in args.sizes.split(",") if s]
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    bytes_elem = 2 if args.dtype == "fp16" else 4
    bytes_per_elem_rw = bytes_elem * 2   # 读 + 写 (element-wise 对称)

    rng = np.random.default_rng(0xC0FFEE)
    summary = []
    print("N,bytes,ms_best,GBps,GFLOPS_elem,correctness_max_abs")

    for N in sizes:
        x_np = (rng.standard_normal(N) * 3.0).astype(
            np.float16 if args.dtype == "fp16" else np.float32
        )
        x_dev = torch.from_numpy(x_np.copy()).npu()

        # correctness 与 numpy reference 比较 (拿第一轮的结果)
        y_dev = gelu_triton(x_dev, block_size=args.block_size)
        y_ref = gelu_reference_numpy(x_np)
        y_cpu = y_dev.cpu().numpy()
        max_abs = float(np.max(np.abs(
            y_cpu.astype(np.float32) - y_ref.astype(np.float32)
        )))
        assert max_abs < 5e-2, f"FAIL correctness N={N} max_abs={max_abs}"

        ms = bench_one(x_dev, None, args.block_size, args.warmup, args.repeats)
        bytes_total = N * bytes_per_elem_rw
        gbps = bytes_total / (ms * 1e-3) / 1e9
        # element-wise FLOPs 估: 每元素约做 7 个 scalar 浮点 (3x mul + 2 add + tanh(~2 mul + 1 div) 估成 8)
        # 保守统一算 8 FLOPs / element
        flops = N * 8
        gflops = flops / (ms * 1e-3) / 1e9
        print(f"{N},{bytes_total},{ms:.4f},{gbps:.2f},{gflops:.1f},{max_abs:.2e}")
        sys.stdout.flush()
        summary.append(dict(N=N, bytes_total=bytes_total, ms_best_ms=round(ms, 4),
                            GBps=round(gbps, 3), GFLOPS=round(gflops, 2),
                            max_abs=max_abs))

    # summary 到 stderr
    obj = dict(benchmark="gelu_triton_ascend",
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
