"""
FlashAttention benchmark —— Triton-Ascend FA2 前向 (非因果), 多种 (H, L, S, D).

用法:
  python3 bench_flash_triton.py [--device 0] [--repeats 20]

输出: CSV 到 stdout + JSON 摘要到 stderr.
Roofline: FA 的 IO 复杂度 O(S·D) 每头每 q 块 (读 KV 一遍) + q/out,
算术强度 ~ O(S) — 带宽受限 (KV 读取主导); 与标准注意力相比省的是
L×S 分数矩阵的 HBM 往返 (本实现根本不物化).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from flash_triton import flash_attention_triton, attention_reference_numpy

HBM_TBPS = 1.6


def main() -> int:
    ap = argparse.ArgumentParser(description="Triton-Ascend FlashAttention 微基准.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.npu.set_device(args.device)

    # (H, L, S, D)
    cases = [
        (2, 1024, 1024, 64),
        (2, 1024, 1024, 128),
        (4, 1024, 1024, 128),
        (8, 1024, 1024, 128),
        (8, 2048, 2048, 128),
        (32, 1024, 1024, 128),
    ]

    rng = np.random.default_rng(0xC0FFEE)
    summary = []
    print("H,L,S,D,kv_bytes,ms_best,GBps,correctness_max_abs")

    for H, L, S, D in cases:
        q_np = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
        k_np = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
        v_np = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)

        ref = attention_reference_numpy(q_np, k_np, v_np)
        q_dev = torch.from_numpy(q_np).npu()
        k_dev = torch.from_numpy(np.ascontiguousarray(k_np)).npu()
        v_dev = torch.from_numpy(np.ascontiguousarray(v_np)).npu()
        out_dev = flash_attention_triton(q_dev, k_dev, v_dev,
                                         BLOCK_M=args.block_m, BLOCK_N=args.block_n)
        out_np = out_dev.cpu().numpy()
        max_abs = float(np.max(np.abs(out_np.astype(np.float32) - ref.astype(np.float32))))
        assert max_abs < 5e-2, f"FAIL correctness H={H} L={L} S={S}"

        for _ in range(args.warmup):
            _ = flash_attention_triton(q_dev, k_dev, v_dev,
                                       BLOCK_M=args.block_m, BLOCK_N=args.block_n)
        torch.npu.synchronize()
        best = float("inf")
        for _ in range(args.repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter_ns()
            _ = flash_attention_triton(q_dev, k_dev, v_dev,
                                       BLOCK_M=args.block_m, BLOCK_N=args.block_n)
            torch.npu.synchronize()
            t1 = time.perf_counter_ns()
            best = min(best, (t1 - t0) * 1e-6)

        kv_bytes = 2 * H * S * D * 2   # K+V 读取, fp16
        gbps = kv_bytes / (best * 1e-3) / 1e9
        print(f"{H},{L},{S},{D},{kv_bytes},{best:.4f},{gbps:.2f},{max_abs:.2e}")
        sys.stdout.flush()
        summary.append(dict(H=H, L=L, S=S, D=D, ms_best=round(best, 4),
                            GBps=round(gbps, 3), max_abs=max_abs))

    obj = dict(benchmark="flash_triton_ascend", SoC="910B2", device_id=args.device,
               block_m=args.block_m, block_n=args.block_n, rows=summary,
               HBM_TBPS_quoted=HBM_TBPS)
    print("\n" + json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
