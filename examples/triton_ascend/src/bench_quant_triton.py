"""
INT8 量化 benchmark —— Triton-Ascend quant+dequant 全链路, 多种 (M, D), 多轮.

用法:
  python3 bench_quant_triton.py [--device 0] [--rows 1024,4096,16384] \
      [--dims 512,4096] [--repeats 20]

输出: CSV 到 stdout + JSON 摘要到 stderr.
Roofline: 每元素读 fp16 x (2B) + 写 int8 q (1B) (+scale 摊销) ≈ 3 B/元素,
带宽受限; dequant 反向同理.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from quant_triton import quant_int8_triton, dequant_int8_triton, quant_int8_reference_numpy

HBM_TBPS = 1.6


def bench_quant(x_npu, warmup: int, repeats: int) -> float:
    import torch
    for _ in range(warmup):
        _ = quant_int8_triton(x_npu)
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        _ = quant_int8_triton(x_npu)
        torch.npu.synchronize()
        t1 = time.perf_counter_ns()
        best = min(best, (t1 - t0) * 1e-6)
    return best


def bench_dequant(q_npu, s_npu, warmup: int, repeats: int) -> float:
    import torch
    for _ in range(warmup):
        _ = dequant_int8_triton(q_npu, s_npu)
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        _ = dequant_int8_triton(q_npu, s_npu)
        torch.npu.synchronize()
        t1 = time.perf_counter_ns()
        best = min(best, (t1 - t0) * 1e-6)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Triton-Ascend INT8 量化微基准.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rows", type=str, default="1024,4096,16384")
    ap.add_argument("--dims", type=str, default="512,4096")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.npu.set_device(args.device)

    rows = [int(s) for s in args.rows.split(",") if s]
    dims = [int(s) for s in args.dims.split(",") if s]
    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    rng = np.random.default_rng(0xC0FFEE)
    summary = []
    print("M,D,quant_ms,dequant_ms,quant_GBps,correctness_roundtrip")

    for M in rows:
        for D in dims:
            x_np = (rng.standard_normal((M, D)) * 2.0).astype(np_dtype)
            x_dev = torch.from_numpy(x_np.copy()).npu()

            # correctness
            q_dev, s_dev = quant_int8_triton(x_dev)
            q_ref, s_ref = quant_int8_reference_numpy(x_np)
            y_dev = dequant_int8_triton(q_dev, s_dev, torch_dtype)
            rt = float(np.max(np.abs(y_dev.cpu().numpy().astype(np.float32) - x_np.astype(np.float32))))
            assert rt <= float(s_ref.max()) + 1e-6, f"roundtrip FAIL M={M} D={D}"

            ms_q = bench_quant(x_dev, args.warmup, args.repeats)
            ms_dq = bench_dequant(q_dev, s_dev, args.warmup, args.repeats)
            # 量化访存: 读 fp16 x (2B) + 写 int8 q (1B); 反量化: 读 int8 + 写 fp16
            gbps_q = (M * D * 3) / (ms_q * 1e-3) / 1e9
            print(f"{M},{D},{ms_q:.4f},{ms_dq:.4f},{gbps_q:.2f},{rt:.4f}")
            sys.stdout.flush()
            summary.append(dict(M=M, D=D, quant_ms=round(ms_q, 4), dequant_ms=round(ms_dq, 4),
                                quant_GBps=round(gbps_q, 3), roundtrip=rt))

    obj = dict(benchmark="quant_triton_ascend", SoC="910B2", device_id=args.device,
               dtype=args.dtype, rows=summary, HBM_TBPS_quoted=HBM_TBPS)
    print("\n" + json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
