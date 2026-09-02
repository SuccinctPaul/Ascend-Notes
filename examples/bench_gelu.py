"""
统一跑三家 (NumPy ground truth / Triton-Ascend / TileLang-Ascend if installed)
GELU 微基准 + Roofline 计算, 输出一份 CSV + 一份 JSON 供 docs 直接贴表.

同时通过 ascend_gelu 二进制跑 Ascend C (教学版标量 kernel) 性能:
  ascend_gelu <N> 的输出我们会解析.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent   # examples/
PY_SRC = ROOT / "python" / "src"
TRITON_SRC = ROOT / "triton_ascend" / "src"
TILE_SRC = ROOT / "tilelang_ascend" / "src"
ASCEND_BUILD = ROOT / "ascend_c" / "build"

# bench_gelu.py 既可以 `python examples/bench_gelu.py` 直接跑,
# 也可以被以任何 cwd 调用, 因此这里先强制把三个 src 路径加进 sys.path.
for p in (PY_SRC, TRITON_SRC, TILE_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from gelu import gelu_reference  # noqa: E402  ground truth


# 910B2 公开硬件参数 (单 chip, 保守值)
#   算力峰 Vector (fp16): 约 280 TFLOPS (取昇腾文档中的 910B2 AI Core Vector MAC 峰值)
#   HBM 带宽: 约 1.6 TB/s (单 chip 910B2 HBM2e 理论带宽)
#   Cube 算力不用, GELU 是 element-wise 只用 Vector.
THEORETICAL_PEAK_TFLOPS_FP16_VECTOR = 280.0
HBM_TBPS = 1.6

# 单元素 GELU (tanh 近似):
#   x^3          -> 2 mul
#   x + c x^3    -> 1 madd
#   s2p*(...)    -> 1 mul
#   tanh(inner)  -> 约 4 mul + 2 add + 1 div  (有理分式近似)
#   (1 + t)      -> 1 add
#   0.5*(1+t)    -> 1 mul
#   x * g        -> 1 mul
# 合计 ≈ 11 FLOPs/元素
FLOPS_PER_ELEMENT = 11


def numpy_bench(sizes, repeats, dtype=np.float32):
    rows = []
    rng = np.random.default_rng(1)
    for N in sizes:
        x = (rng.standard_normal(N) * 3.0).astype(dtype)
        # warmup
        for _ in range(3):
            _ = gelu_reference(x)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            _ = gelu_reference(x)
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) * 1e-6
            if ms < best:
                best = ms
        bytes_per_elem = np.dtype(dtype).itemsize * 2
        bw = N * bytes_per_elem / (best * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best * 1e-3) / 1e9
        rows.append(dict(N=N, dtype=np.dtype(dtype).name,
                         ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         AI="—"))  # numpy 跑 host
    return rows


def triton_bench(sizes, repeats, device, dtype="fp16", block_size=1024):
    try:
        import torch
    except Exception as e:
        return [dict(error=f"torch not found: {e}")]
    try:
        from gelu_triton import gelu_triton, gelu_reference_numpy
    except Exception as e:
        return [dict(error=f"triton gelu import: {e}")]
    torch.npu.set_device(device)
    dt = torch.float16 if dtype == "fp16" else torch.float32
    bytes_elem = 2 if dtype == "fp16" else 4
    rng = np.random.default_rng(2)
    rows = []
    for N in sizes:
        x_np = (rng.standard_normal(N) * 3.0).astype(np.float16 if dtype == "fp16" else np.float32)
        x_dev = torch.from_numpy(x_np.copy()).npu()
        # correctness
        y_dev = gelu_triton(x_dev, block_size=block_size)
        ref = gelu_reference_numpy(x_np)
        max_abs = float(np.max(np.abs(
            y_dev.cpu().numpy().astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
        # warmup
        for _ in range(3):
            _ = gelu_triton(x_dev, block_size=block_size)
        torch.npu.synchronize()
        best = float("inf")
        for _ in range(repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter_ns()
            _ = gelu_triton(x_dev, block_size=block_size)
            torch.npu.synchronize()
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) * 1e-6
            if ms < best:
                best = ms
        bw = N * bytes_elem * 2 / (best * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best * 1e-3) / 1e9
        rows.append(dict(N=N, dtype=dtype, ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         max_abs=max_abs))
    return rows


def torch_npu_gelu_bench(sizes, repeats, device, dtype="fp16"):
    """PyTorch 自带 F.gelu(approximate='tanh') on NPU —— 内部走 TBE/TIK 融合 kernel.

    用作 "TBE / TIK DSL 等价代表", 作为四家中的第一家 (Python/TorchScript -> 库级 kernel).
    """
    try:
        import torch
    except Exception as e:
        return [dict(error=f"torch not found: {e}")]
    torch.npu.set_device(device)
    dt = torch.float16 if dtype == "fp16" else torch.float32
    bytes_elem = 2 if dtype == "fp16" else 4
    rng = np.random.default_rng(7)
    rows = []
    for N in sizes:
        x_np = (rng.standard_normal(N) * 3.0).astype(
            np.float16 if dtype == "fp16" else np.float32
        )
        x_dev = torch.from_numpy(x_np.copy()).to(dt).npu()
        # correctness vs numpy ref
        import torch.nn.functional as F
        y_dev = F.gelu(x_dev, approximate="tanh")
        ref = gelu_reference(x_np.astype(np.float32))
        max_abs = float(np.max(np.abs(
            y_dev.detach().cpu().numpy().astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
        # warmup
        for _ in range(3):
            _ = F.gelu(x_dev, approximate="tanh")
        torch.npu.synchronize()
        best = float("inf")
        for _ in range(repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter_ns()
            _ = F.gelu(x_dev, approximate="tanh")
            torch.npu.synchronize()
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) * 1e-6
            if ms < best:
                best = ms
        bw = N * (bytes_elem * 2) / (best * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best * 1e-3) / 1e9
        rows.append(dict(N=N, dtype=dtype, ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         max_abs=max_abs))
    return rows


def tilelang_bench(sizes, repeats):
    try:
        import tilelang  # noqa: F401
        from gelu_tilelang import gelu_tilelang, gelu_reference_numpy
    except Exception as e:
        return [dict(error=f"tilelang not available: {e}")]
    rng = np.random.default_rng(3)
    rows = []
    for N in sizes:
        x = (rng.standard_normal(N) * 3.0).astype(np.float16)
        # correctness
        y = gelu_tilelang(x)
        ref = gelu_reference_numpy(x)
        max_abs = float(np.max(np.abs(
            y.astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
        # warmup + timing
        for _ in range(3):
            _ = gelu_tilelang(x)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            _ = gelu_tilelang(x)
            t1 = time.perf_counter_ns()
            ms = (t1 - t0) * 1e-6
            if ms < best:
                best = ms
        bw = N * 4 / (best * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best * 1e-3) / 1e9
        rows.append(dict(N=N, dtype="fp16", ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         max_abs=max_abs))
    return rows


def ascend_c_bench(sizes, repeats, which="prod"):
    """调用 ascend_gelu (PROD DataCopy tile 版本) 或 ascend_gelu_scalar (教学 scalar 地板)
    并解析输出. N 不支持时跳过. 对非 PASS 结果保留, 用于文档中 "scalar 地板性能" 对照."""
    binary = "ascend_gelu" if which == "prod" else "ascend_gelu_scalar"
    dtype_tag = ("fp16(PROD DataCopy tile)" if which == "prod"
                 else "fp16(教学 scalar 地板对照)")
    bin_path = ASCEND_BUILD / binary
    if not bin_path.exists():
        return [dict(error=f"{binary} not built at {bin_path}")]
    rows = []
    for N in sizes:
        best_ms = float("inf")
        max_abs = None
        last_pass = None
        for _ in range(repeats):
            p = subprocess.run([str(bin_path), str(N)], capture_output=True, text=True,
                               timeout=240)
            out = (p.stdout or "") + (p.stderr or "")
            m = [ln for ln in out.splitlines() if "kernel ms" in ln]
            a = [ln for ln in out.splitlines() if "max_abs_err" in ln]
            r = [ln for ln in out.splitlines() if "result" in ln]
            if m:
                ms = float(m[0].split("=")[1].strip().split()[0])
                best_ms = min(best_ms, ms)
            if a:
                max_abs = float(a[0].split("=")[1].strip())
            if r:
                last_pass = "PASS" in r[0]
        if best_ms == float("inf"):
            rows.append(dict(N=N, error="no kernel ms parsed"))
            continue
        bw = N * 4 / (best_ms * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best_ms * 1e-3) / 1e9
        rows.append(dict(N=N, dtype=dtype_tag, ms_best_ms=round(best_ms, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         max_abs=max_abs, pass_flag=last_pass,
                         which=which))
    return rows


def roofline_summary(rows_np, rows_torch, rows_tri, rows_tile, rows_asc):
    """给每个成功的数据点计算 intensity / ratio_to_bw / ratio_to_peak."""
    out = []
    for label, rows in [
        ("NumPy_ref_CPU", rows_np),
        ("TorchNPU-Native(TBE/TIK)_910B2", rows_torch),
        ("Triton-Ascend_910B2", rows_tri),
        ("TileLang-Ascend_910B2", rows_tile),
        ("AscendC_910B2_scalar", rows_asc),
    ]:
        for r in rows:
            if "error" in r or r.get("pass_flag") is False:
                continue
            N = r["N"]
            GFLOPS = r.get("GFLOPS")
            GBps   = r.get("GBps")
            if GFLOPS is None or GBps is None or GBps <= 0:
                continue
            I = GFLOPS / GBps if GBps > 0 else 0  # FLOPs/Byte
            peak_gflops_sat_by_bw = HBM_TBPS * 1000 * I  # TB/s * 1000 GB/TB * I = GFLOPS/s
            sat_by_peak = THEORETICAL_PEAK_TFLOPS_FP16_VECTOR * 1000  # TFLOPS * 1000
            predicted_perf = min(peak_gflops_sat_by_bw, sat_by_peak)
            eff_wrt_predicted = GFLOPS / predicted_perf if predicted_perf > 0 else float("nan")
            bw_util = GBps / (HBM_TBPS * 1000)
            compute_util = GFLOPS / sat_by_peak
            out.append(dict(label=label, N=N, dtype=r.get("dtype", "?"),
                            GFLOPS=GFLOPS, GBps=GBps,
                            intensity_FLOP_per_Byte=round(I, 3),
                            roofline_predicted_GFLOPS=round(predicted_perf, 1),
                            efficiency_wrt_roofline=round(eff_wrt_predicted, 3),
                            HBM_util_pct=round(bw_util * 100, 2),
                            Vector_peak_util_pct=round(compute_util * 100, 3)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="跑 NumPy / Triton-Ascend / TileLang-Ascend / Ascend C 教学版 "
                    "GELU 微基准 + Roofline 摘要. "
                    "注意: 若设置了 ASCEND_RT_VISIBLE_DEVICES, --device 指的是*可见域内逻辑编号* "
                    "(例如 ASCEND_RT_VISIBLE_DEVICES=2 时 --device=0 对应物理 2).")
    ap.add_argument("--sizes", default="4096,65536,1048576,8388608,33554432,67108864,134217728")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--device", type=int, default=0,
                    help="NPU 逻辑设备号 (ASCEND_RT_VISIBLE_DEVICES 重映射后). 建议配合: "
                         "ASCEND_RT_VISIBLE_DEVICES=<phy_id> python3 bench_gelu.py --device 0")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--run", default="all",
                    help="csv of tags to run: numpy,torch,triton,tilelang,ascendc")
    ap.add_argument("--out", default="", help="write JSON summary here")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x]
    tags = set(args.run.split(","))
    run_all = "all" in tags

    result = dict(
        SoC="910B2",
        device_id_used_for_triton=args.device,
        THEORETICAL_PEAK_TFLOPS_FP16_VECTOR=THEORETICAL_PEAK_TFLOPS_FP16_VECTOR,
        HBM_TBPS_QUOTED=HBM_TBPS,
        FLOPS_PER_ELEMENT=FLOPS_PER_ELEMENT,
    )

    if run_all or "numpy" in tags:
        result["numpy_cpu_fp32"] = numpy_bench(sizes, args.repeats, dtype=np.float32)
    if run_all or "torch" in tags:
        result["torch_npu_fp16"] = torch_npu_gelu_bench(
            sizes, args.repeats, args.device, dtype="fp16"
        )
    if run_all or "triton" in tags:
        result["triton_npu_fp16"] = triton_bench(sizes, args.repeats, args.device,
                                                  dtype="fp16", block_size=args.block_size)
    if run_all or "tilelang" in tags:
        try:
            result["tilelang_npu_fp16"] = tilelang_bench(sizes, args.repeats)
        except Exception as e:
            print(f"[SKIP] tilelang_bench failed: {type(e).__name__}: {e}")
            result["tilelang_npu_fp16"] = None
    if run_all or "ascendc" in tags:
        # 生产版 (DataCopy tile) — 性能与正确性主指标
        result["ascendc_npu_fp16_prod"]   = ascend_c_bench(sizes, args.repeats, which="prod")
        # 教学版 scalar (bisheng CANN 9.0 scalar 地板性能对照, 故意多 blocks FAIL)
        result["ascendc_npu_fp16_scalar"] = ascend_c_bench(sizes, args.repeats, which="scalar")

    rl = roofline_summary(
        result.get("numpy_cpu_fp32") or [],
        result.get("torch_npu_fp16") or [],
        result.get("triton_npu_fp16") or [],
        result.get("tilelang_npu_fp16") or [],
        (result.get("ascendc_npu_fp16_prod") or []) + (result.get("ascendc_npu_fp16_scalar") or []),
    )
    result["roofline_points"] = rl

    txt = json.dumps(result, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
