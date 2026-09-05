"""
TileLang-Ascend RMSNorm / RoPE 微基准。

用法:
  python3 bench_tilelang_ops.py [--rows 64,256,1024] [--dims 512,1024,4096] [--repeats 5]

说明:
  TileLang 教学版 kernel 是 "一个 AI Core 处理一整行" 的 1D kernel, 封装层对
  M 行做 Python 循环逐行调用 (与 softmax_tilelang 同构), 因此:
    - 每个 kernel call 有固定的 Python/launch 开销, 对小行数结果影响大;
    - ms 为 "M 行全部算完" 的端到端 wall time (含 npu 同步), 与 triton 版
      bench (一次 kernel launch 覆盖全部行) 不直接可比, 是本教学实现的真实成本。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from rmsnorm_tilelang import rmsnorm_tilelang, rmsnorm_reference_numpy
from rope_tilelang import rope_tilelang, rope_reference_numpy
from quant_tilelang import quant_int8_tilelang, dequant_int8_tilelang, quant_int8_reference_numpy
from gqa_tilelang import gqa_decode_tilelang, gqa_decode_reference_numpy
from flash_tilelang import flash_attention_tilelang, attention_reference_numpy


def _bench_loop(fn, warmup: int, repeats: int) -> float:
    """返回最快一轮 wall ms (整段调用, 含同步)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        best = min(best, (t1 - t0) * 1e-6)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="TileLang-Ascend RMSNorm/RoPE 微基准.")
    ap.add_argument("--rows", type=str, default="16,64,256")
    ap.add_argument("--dims", type=str, default="512,1024,4096")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    rows = [int(s) for s in args.rows.split(",") if s]
    dims = [int(s) for s in args.dims.split(",") if s]

    rng = np.random.default_rng(0xC0FFEE)
    summary = {"rmsnorm": [], "rope": []}

    # ---------------- RMSNorm ----------------
    print("op,rmsnorm: M,D,ms_best,ms_per_row,GBps,correctness_max_abs")
    for M in rows:
        for D in dims:
            x = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
            gamma = rng.uniform(0.5, 2.0, D).astype(np.float16)
            # correctness 一次
            y = rmsnorm_tilelang(x, gamma)
            ref = rmsnorm_reference_numpy(x, gamma)
            max_abs = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
            assert max_abs < 5e-2, f"rmsnorm FAIL M={M} D={D} max_abs={max_abs}"
            ms = _bench_loop(lambda: rmsnorm_tilelang(x, gamma),
                             args.warmup, args.repeats)
            bytes_total = M * D * 2 * 2  # fp16 读 x + 写 y
            gbps = bytes_total / (ms * 1e-3) / 1e9
            print(f"rmsnorm,{M},{D},{ms:.3f},{ms / M:.4f},{gbps:.2f},{max_abs:.2e}")
            sys.stdout.flush()
            summary["rmsnorm"].append(dict(M=M, D=D, ms=round(ms, 3),
                                           ms_per_row=round(ms / M, 4),
                                           GBps=round(gbps, 3), max_abs=max_abs))

    # ---------------- RoPE ----------------
    print("op,rope: T,D,ms_best,ms_per_row,GBps,correctness_max_abs")
    for T_ in rows:
        for D in dims:
            x = (rng.standard_normal((T_, D)) * 2.0).astype(np.float16)
            pos = rng.integers(0, 8192, size=T_)
            y = rope_tilelang(x, pos)
            ref = rope_reference_numpy(x, pos)
            max_abs = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
            assert max_abs < 5e-2, f"rope FAIL T={T_} D={D} max_abs={max_abs}"
            ms = _bench_loop(lambda: rope_tilelang(x, pos),
                             args.warmup, args.repeats)
            bytes_total = T_ * D * 2 * 3  # fp16 读 x + cos/sin + 写 y
            gbps = bytes_total / (ms * 1e-3) / 1e9
            print(f"rope,{T_},{D},{ms:.3f},{ms / T_:.4f},{gbps:.2f},{max_abs:.2e}")
            sys.stdout.flush()
            summary["rope"].append(dict(T=T_, D=D, ms=round(ms, 3),
                                        ms_per_row=round(ms / T_, 4),
                                        GBps=round(gbps, 3), max_abs=max_abs))

    # ---------------- INT8 量化 (quant+dequant 全链路) ----------------
    print("op,quant: M,D,ms_best,ms_per_row,GBps,correctness_roundtrip")
    summary["quant"] = []
    for M in rows:
        for D in dims:
            x = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
            q, scale = quant_int8_tilelang(x)
            q_ref, scale_ref = quant_int8_reference_numpy(x)
            y = dequant_int8_tilelang(q, scale)
            rt = float(np.max(np.abs(y.astype(np.float32) - x.astype(np.float32))))
            assert rt <= float(scale_ref.max()) + 1e-6, f"quant FAIL M={M} D={D}"
            ms = _bench_loop(lambda: dequant_int8_tilelang(*quant_int8_tilelang(x)),
                             args.warmup, args.repeats)
            print(f"quant,{M},{D},{ms:.3f},{ms / M:.4f},{M * D * 3 / (ms * 1e-3) / 1e9:.2f},{rt:.4f}")
            sys.stdout.flush()
            summary["quant"].append(dict(M=M, D=D, ms=round(ms, 3),
                                         ms_per_row=round(ms / M, 4), roundtrip=rt))

    # ---------------- GQA 解码 (小档位) ----------------
    print("op,gqa: Hq,Hkv,S,D,ms_best,ms_per_row,correctness_max_abs")
    summary["gqa"] = []
    for Hq, Hkv, S, D in [(4, 2, 64, 64), (8, 2, 128, 64)]:
        q = (rng.standard_normal((Hq, D)) * 1.5).astype(np.float16)
        k = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        v = (rng.standard_normal((Hkv, S, D)) * 1.5).astype(np.float16)
        out = gqa_decode_tilelang(q, k, v)
        ref = gqa_decode_reference_numpy(q, k, v)
        max_abs = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
        assert max_abs < 5e-2, f"gqa FAIL {Hq},{Hkv},{S},{D}"
        ms = _bench_loop(lambda: gqa_decode_tilelang(q, k, v), args.warmup, args.repeats)
        print(f"gqa,{Hq},{Hkv},{S},{D},{ms:.3f},{ms / Hq:.4f},{max_abs:.2e}")
        sys.stdout.flush()
        summary["gqa"].append(dict(Hq=Hq, Hkv=Hkv, S=S, D=D, ms=round(ms, 3), max_abs=max_abs))

    # ---------------- FlashAttention (小档位) ----------------
    print("op,flash: H,L,S,D,ms_best,ms_per_row,correctness_max_abs")
    summary["flash"] = []
    for H, L, S, D in [(1, 8, 64, 64), (2, 8, 64, 64)]:
        q = (rng.standard_normal((H, L, D)) * 1.5).astype(np.float16)
        k = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
        v = (rng.standard_normal((H, S, D)) * 1.5).astype(np.float16)
        out = flash_attention_tilelang(q, k, v)
        ref = attention_reference_numpy(q, k, v)
        max_abs = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
        assert max_abs < 5e-2, f"flash FAIL {H},{L},{S},{D}"
        ms = _bench_loop(lambda: flash_attention_tilelang(q, k, v), args.warmup, args.repeats)
        print(f"flash,{H},{L},{S},{D},{ms:.3f},{ms / (H * L):.4f},{max_abs:.2e}")
        sys.stdout.flush()
        summary["flash"].append(dict(H=H, L=L, S=S, D=D, ms=round(ms, 3), max_abs=max_abs))

    obj = dict(benchmark="tilelang_ascend_ops", SoC="910B2",
               note="per-row serial kernels, python loop over rows (teaching impl)",
               repeats=args.repeats, rows=summary)
    print("\n" + json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
