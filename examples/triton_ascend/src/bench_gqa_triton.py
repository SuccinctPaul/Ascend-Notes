"""
GQA 解码注意力 benchmark —— Triton-Ascend, 多种 (Hq, Hkv, S, D), 多轮.

用法:
  python3 bench_gqa_triton.py [--device 0] [--repeats 20]

输出: CSV 到 stdout + JSON 摘要到 stderr.
Roofline: 解码一步读 KV cache (2·Hkv·S·D·2B) + q/out (2·Hq·D·2B),
算术强度 ~ 2·S FLOP/元素, 小 S 下带宽受限 (KV cache 读取主导).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from gqa_triton import gqa_decode_triton, gqa_decode_reference_numpy

HBM_TBPS = 1.6


def main() -> int:
    ap = argparse.ArgumentParser(description="Triton-Ascend GQA 解码微基准.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-s", type=int, default=128)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    import torch
    torch.npu.set_device(args.device)

    # (Hq, Hkv, S, D) — 常见模型配置档位
    cases = [
        (8, 8, 1024, 128),    # 7B MHA, 1K ctx
        (32, 8, 1024, 128),   # 7B GQA (LLaMA-3 8B), 1K ctx
        (32, 8, 4096, 128),   # 7B GQA, 4K ctx
        (32, 8, 8192, 128),   # 7B GQA, 8K ctx
        (8, 1, 4096, 64),     # MQA 小模型
    ]

    rng = np.random.default_rng(0xC0FFEE)
    summary = []
    print("Hq,Hkv,S,D,kv_bytes,ms_best,GBps,correctness_max_abs")

    for Hq, Hkv, S, D in cases:
        q_np = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
        k_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        v_np = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)

        ref = gqa_decode_reference_numpy(q_np, k_np, v_np)
        q_dev = torch.from_numpy(q_np).npu()
        k_dev = torch.from_numpy(np.ascontiguousarray(k_np)).npu()
        v_dev = torch.from_numpy(np.ascontiguousarray(v_np)).npu()
        out_dev = gqa_decode_triton(q_dev, k_dev, v_dev, BLOCK_S=args.block_s)
        out_np = out_dev.cpu().numpy()
        max_abs = float(np.max(np.abs(out_np.astype(np.float32) - ref.astype(np.float32))))
        assert max_abs < 5e-2, f"FAIL correctness Hq={Hq} S={S}"

        for _ in range(args.warmup):
            _ = gqa_decode_triton(q_dev, k_dev, v_dev, BLOCK_S=args.block_s)
        torch.npu.synchronize()
        best = float("inf")
        for _ in range(args.repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter_ns()
            _ = gqa_decode_triton(q_dev, k_dev, v_dev, BLOCK_S=args.block_s)
            torch.npu.synchronize()
            t1 = time.perf_counter_ns()
            best = min(best, (t1 - t0) * 1e-6)

        kv_bytes = 2 * Hkv * S * D * 2   # K+V cache, fp16
        gbps = kv_bytes / (best * 1e-3) / 1e9
        print(f"{Hq},{Hkv},{S},{D},{kv_bytes},{best:.4f},{gbps:.2f},{max_abs:.2e}")
        sys.stdout.flush()
        summary.append(dict(Hq=Hq, Hkv=Hkv, S=S, D=D, ms_best=round(best, 4),
                            GBps=round(gbps, 3), max_abs=max_abs))

    obj = dict(benchmark="gqa_triton_ascend", SoC="910B2", device_id=args.device,
               block_s=args.block_s, rows=summary, HBM_TBPS_quoted=HBM_TBPS)
    print("\n" + json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
