"""
统一跑四家 (NumPy ground truth / PyTorch NPU native / Triton-Ascend /
TileLang-Ascend if installed / Ascend C scalar+production)
GELU 微基准 + Roofline 计算, 输出一份 JSON + 一份控制台汇总.

命令行约定 (对齐项目工程规范):
  --which  scalar|prod|both   (仅作用于 ascendc 分支; 默认 both)
  --run    csv of tags:       numpy,torch,triton,tilelang,ascendc (默认 all)
  --sizes  逗号分隔 N 列表     (默认 7 档, 从 64K 到 128M)
  --repeats                   (默认 15, 每档取 best ms)
  --device N                  (NPU 逻辑设备号, 配合 ASCEND_RT_VISIBLE_DEVICES 使用)
  --out path.json             (写入完整 JSON 结果, 供文档 §8 贴表)

Roofline 指标 (对齐 docs/perf/01-roofline-perf-model.md §1 + 项目内存约定):
  每档 N 都会对成功点计算:
    - intensity_FLOP_per_Byte  (实测 GFLOPS / 实测 GBps, 即工作点的真实 I)
    - roofline_predicted_GFLOPS(min(BW*I, 向量峰值), 即该点的屋顶上限)
    - efficiency_wrt_roofline  (实测 GFLOPS / predicted, 即"离屋顶多近")
    - HBM_util_pct             (实测 GBps / HBM_TBPS*1000, 带宽利用率 %)
    - Vector_peak_util_pct     (实测 GFLOPS / 峰值 GFLOPS, 算力利用率 %)
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


# =============================================================================
# 硬件常量 —— 910B2 公开规格的保守、可复现估计 (见 perf/00-npu-peak-flops-calculation.md)
#   Vector fp16 峰值: ~280 TFLOPS (24 core * 1.5GHz * 8192/cycle FP16 向量化估计)
#   HBM 带宽:        ~1.6 TB/s  (910B2 HBM2e 标称值, 实际单向读约 1.2 TB/s)
# =============================================================================
THEORETICAL_PEAK_TFLOPS_FP16_VECTOR = 280.0
HBM_TBPS = 1.6

# 单元素 GELU (tanh 近似) FLOPs 计数 (与之前版本保持一致, 11 FLOP/elem):
#   x^2            1 mul
#   x^3            1 mul  (=x^2 * x)
#   0.044715*x^3   1 mul
#   x + ...        1 add
#   sqrt(2/pi)*t   1 mul
#   tanh(inner)    ~4 mul + 2 add + 1 div (Pade[7,7] 有理分式近似)
#   1 + tanh       1 add
#   0.5*(1+tanh)   1 mul
#   x * gate       1 mul
# 合计: = 8 mul + 4 add + 1 div ≈ 13, 但行业经验按 ~11 计.
FLOPS_PER_ELEMENT = 11


# =============================================================================
# 1. NumPy (CPU, 正确性基准 + 纯 CPU 性能基线)
# =============================================================================
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
        bytes_per_elem = np.dtype(dtype).itemsize * 2  # 读 x + 写 y
        bw = N * bytes_per_elem / (best * 1e-3) / 1e9
        gflops = N * FLOPS_PER_ELEMENT / (best * 1e-3) / 1e9
        rows.append(dict(N=N, dtype=np.dtype(dtype).name,
                         ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         AI="—"))  # numpy 跑 host, 不进 NPU Roofline 体系
    return rows


# =============================================================================
# 2. PyTorch NPU 自带 F.gelu(approximate='tanh') —— 代表 TBE/TIK 官方库实现
# =============================================================================
def torch_npu_gelu_bench(sizes, repeats, device, dtype="fp16"):
    try:
        import torch
    except Exception as e:
        return [dict(error=f"torch not found: {e}")]
    torch.npu.set_device(device)
    bytes_elem = 2 if dtype == "fp16" else 4
    dt = torch.float16 if dtype == "fp16" else torch.float32
    rng = np.random.default_rng(7)
    rows = []
    for N in sizes:
        x_np = (rng.standard_normal(N) * 3.0).astype(
            np.float16 if dtype == "fp16" else np.float32
        )
        x_dev = torch.from_numpy(x_np.copy()).to(dt).npu()
        import torch.nn.functional as F
        y_dev = F.gelu(x_dev, approximate="tanh")
        ref = gelu_reference(x_np.astype(np.float32))
        max_abs = float(np.max(np.abs(
            y_dev.detach().cpu().numpy().astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
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


# =============================================================================
# 3. Triton-Ascend
# =============================================================================
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
    bytes_elem = 2 if dtype == "fp16" else 4
    rng = np.random.default_rng(2)
    rows = []
    for N in sizes:
        x_np = (rng.standard_normal(N) * 3.0).astype(np.float16 if dtype == "fp16" else np.float32)
        x_dev = torch.from_numpy(x_np.copy()).npu()
        y_dev = gelu_triton(x_dev, block_size=block_size)
        ref = gelu_reference_numpy(x_np)
        max_abs = float(np.max(np.abs(
            y_dev.cpu().numpy().astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
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


# =============================================================================
# 4. TileLang-Ascend
# =============================================================================
_TL_ERR_HINT = (
    "\n[HINT] TileLang-Ascend 排障总入口见项目文档: "
    "docs/ops/05-gelu.md §8 TileLang 验证步骤 & 常见坑."
    " 常见错误分类 (#TL-1..#TL-5):"
    "\n  #TL-1 No registered target detector for 'llvm --keys=ascend' → "
    "未装 tilelang-ascend wheel."
    "\n  #TL-2 Unsupported scope: src=global, dst=local → "
    "UB 缓冲必须用 T.alloc_ub / T.alloc_L1, 不能用默认 T.alloc_local."
    "\n  #TL-3 Unresolved call Op(tir.tanh|exp|sigmoid) → "
    "只能用 T.ascend_tile.<op>(dst, src, ...) 的 buffer 级 intrinsic; "
    "add/mul 可以接受 float scalar, sub/div 必须给 Buffer."
    "\n  #TL-4 E39007 Inner_Error_Device_Subprocess_Startup_Timeout / "
    "rtSetDevice err 507033 → CANN 容器 HDC 挂了, "
    "执行 `npu-smi set -t reset -i 0 -c 0` 或联系管理员重启 Host 侧 device daemon."
    "\n  #TL-5 NameError: name 'D' is not defined → "
    "TileLang 解析闭包参数注解 bug, 需把 D/BLOCK/dtype 注入 sys.modules[__name__].__dict__."
)


def _annotate_tl_error(e: Exception) -> Exception:
    """把常见 TileLang-ascend 错误附上排障 hint, 不吞原始 traceback."""
    msg = str(e)
    if any(k in msg for k in ("Unsupported scope", "Unresolved call Op(tir.",
                               "No registered target detector",
                               "Device_Subprocess_Startup_Timeout",
                               "507033", "don't know how to convert type",
                               "NameError: name '",
                               "expected Object but got str",  # #TL-5a: future annotations
                               "DiagnosticError")):
        try:
            e.args = (f"{msg}{_TL_ERR_HINT}", *e.args[1:])
        except Exception:
            pass
    return e


def tilelang_bench(sizes, repeats):
    try:
        import tilelang  # noqa: F401
        from gelu_tilelang import gelu_tilelang, gelu_reference_numpy
    except Exception as e:
        _annotate_tl_error(e)
        return [dict(error=f"tilelang not available: {e}{_TL_ERR_HINT}")]
    rng = np.random.default_rng(3)
    rows = []
    for N in sizes:
        x = (rng.standard_normal(N) * 3.0).astype(np.float16)
        try:
            y = gelu_tilelang(x)
        except Exception as e:
            _annotate_tl_error(e)
            rows.append(dict(N=N, error=f"run: {type(e).__name__}: {e}"))
            continue
        ref = gelu_reference_numpy(x)
        max_abs = float(np.max(np.abs(
            y.astype(np.float32) - ref.astype(np.float32)
        )))
        if max_abs > 5e-2:
            rows.append(dict(N=N, error=f"correctness fail max_abs={max_abs}"))
            continue
        try:
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
        except Exception as e:
            _annotate_tl_error(e)
            rows.append(dict(N=N, dtype="fp16", max_abs=max_abs,
                             error=f"timing: {type(e).__name__}: {e}"))
            continue
        rows.append(dict(N=N, dtype="fp16", ms_best_ms=round(best, 4),
                         GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                         max_abs=max_abs))
    return rows


# =============================================================================
# 5. Ascend C (教学版标量 + 生产版 Vector tile 双分支)
#    --which=scalar  ascend_gelu_scalar     (地板性能基线)
#    --which=prod    ascend_gelu            (LocalTensor tile+DataCopy+Vector)
#    --which=both    都跑
# =============================================================================
def _run_ascendc_binary(bin_path: Path, N: int, repeats: int, extra_arg: str = ""):
    """通用: 跑一个 ascendc binary N 次, 返回 (best_ms, max_abs, pass_flag)."""
    if not bin_path.exists():
        return None
    best_ms = float("inf")
    max_abs = None
    pass_flag = None
    for _ in range(repeats):
        cmd = [str(bin_path), str(N)]
        if extra_arg:
            cmd.append(extra_arg)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        m_lines = [ln for ln in out.splitlines() if "kernel ms" in ln]
        a_lines = [ln for ln in out.splitlines() if "max_abs_err" in ln]
        r_lines = [ln for ln in out.splitlines() if "result" in ln]
        if m_lines:
            # e.g. "kernel ms = 0.1234 (含同步，仅粗测)"
            parts = m_lines[0].split("=")
            ms = float(parts[1].strip().split()[0])
            best_ms = min(best_ms, ms)
        if a_lines:
            max_abs = float(a_lines[0].split("=")[1].strip())
        if r_lines:
            pass_flag = "PASS" in r_lines[0]
    return dict(best_ms=None if best_ms == float("inf") else best_ms,
                max_abs=max_abs, pass_flag=pass_flag)


def ascend_c_bench(sizes, repeats, which: str = "both"):
    """Ascend C 分支: which ∈ {scalar, prod, both}.
    prod 用 ascend_gelu (默认生产版 Vector tile kernel);
    scalar 用 ascend_gelu_scalar (独立编译的标量地板版).
    """
    bin_prod   = ASCEND_BUILD / "ascend_gelu"
    bin_scalar = ASCEND_BUILD / "ascend_gelu_scalar"

    to_run = []  # list of (label, bin_path, extra_arg)
    if which in ("prod", "both"):
        if bin_prod.exists():
            to_run.append(("ascendc_production", bin_prod, ""))
        else:
            to_run.append(("ascendc_production", None, ""))
    if which in ("scalar", "both"):
        if bin_scalar.exists():
            to_run.append(("ascendc_scalar", bin_scalar, ""))
        else:
            # Fallback: 若未单独编译 scalar binary, 尝试调用 ascend_gelu + runtime
            # "scalar" 参数 (要求 ascend_gelu 同时链接了两个 stub lib).
            if bin_prod.exists():
                to_run.append(("ascendc_scalar_via_prod", bin_prod, "scalar"))
            else:
                to_run.append(("ascendc_scalar", None, ""))

    result_rows: dict[str, list] = {}
    for label, binary, extra in to_run:
        rows = []
        if binary is None:
            rows.append(dict(error=f"{label}: binary not found at build dir"))
            result_rows[label] = rows
            continue
        for N in sizes:
            outcome = _run_ascendc_binary(binary, N, repeats, extra_arg=extra)
            if outcome is None:
                rows.append(dict(N=N, error="no output parsed from binary"))
                continue
            if outcome["best_ms"] is None:
                rows.append(dict(N=N, error="no kernel ms parsed", pass_flag=outcome["pass_flag"]))
                continue
            best_ms = outcome["best_ms"]
            bw = N * 4 / (best_ms * 1e-3) / 1e9          # fp16: 读 2B + 写 2B = 4B/elem
            gflops = N * FLOPS_PER_ELEMENT / (best_ms * 1e-3) / 1e9
            row = dict(N=N, dtype="fp16",
                       ms_best_ms=round(best_ms, 4),
                       GBps=round(bw, 3), GFLOPS=round(gflops, 2),
                       max_abs=outcome["max_abs"],
                       pass_flag=outcome["pass_flag"])
            rows.append(row)
        result_rows[label] = rows
    return result_rows


# =============================================================================
# 6. Roofline 汇总: 对每条成功数据点计算 intensity / 预测上限 / 两个 util%
# =============================================================================
def roofline_summary(**named_rowsets):
    """传入 `label_name=rows_dict` 或 `label_name=rows_list`; 递归压平, 统一计算.

    返回每条数据点的 Roofline 分析 dict 列表 (与 bench_gelu_full.json 结构兼容).
    """
    peak_gflops = THEORETICAL_PEAK_TFLOPS_FP16_VECTOR * 1000   # 280_000 GFLOPS
    hbm_gbps    = HBM_TBPS * 1000                              # 1600 GB/s
    out = []
    for label, rows in named_rowsets.items():
        if rows is None:
            continue
        if isinstance(rows, dict):
            # ascendc_bench 返回的嵌套 dict: 把 sub-label 拼上
            for sub, subrows in rows.items():
                subrows = subrows or []
                combined_label = f"{label}_{sub}"
                out.extend(roofline_summary(**{combined_label: subrows}))
            continue
        # list of rows
        for r in rows:
            if not isinstance(r, dict) or "error" in r or r.get("pass_flag") is False:
                continue
            N = r.get("N")
            GFLOPS = r.get("GFLOPS")
            GBps   = r.get("GBps")
            if N is None or GFLOPS is None or GBps is None or GBps <= 0:
                continue
            I = GFLOPS / GBps                                     # FLOPs/Byte
            bw_bound_gflops   = hbm_gbps * I                       # 带宽墙
            comp_bound_gflops = peak_gflops                       # 算力天花板
            predicted_perf    = min(bw_bound_gflops, comp_bound_gflops)
            eff_wrt_roofline  = GFLOPS / predicted_perf if predicted_perf > 0 else float("nan")
            hbm_util          = GBps / hbm_gbps * 100              # %
            vec_util          = GFLOPS / peak_gflops * 100         # %
            out.append(dict(label=label, N=N, dtype=r.get("dtype", "?"),
                            GFLOPS=GFLOPS, GBps=GBps,
                            intensity_FLOP_per_Byte=round(I, 4),
                            roofline_predicted_GFLOPS=round(predicted_perf, 2),
                            efficiency_wrt_roofline=round(eff_wrt_roofline, 4),
                            HBM_util_pct=round(hbm_util, 3),
                            Vector_peak_util_pct=round(vec_util, 4)))
    return out


# =============================================================================
# 7. 控制台打印简表 (不依赖 JSON, 肉眼快速判断)
# =============================================================================
def _print_table(title, rows):
    if not rows:
        print(f"\n[ {title} ]  — (无数据)")
        return
    keys = [k for k in ["N", "dtype", "ms_best_ms", "GBps", "GFLOPS", "max_abs", "error", "pass_flag"]
            if any(k in r for r in rows)]
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    hdr = "  ".join(f"{k:>14s}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _fmt(k, v):
            if isinstance(v, float):
                return f"{v:>14.4g}"
            s = str(v) if v is not None else "—"
            return f"{s:>14s}"[:14]
        print("  ".join(_fmt(k, r.get(k)) for k in keys))


def _print_roofline(points):
    if not points:
        return
    print(f"\n{'='*78}")
    print(f"  Roofline 分析 (Peak Vector fp16 = {THEORETICAL_PEAK_TFLOPS_FP16_VECTOR} TFLOPS, "
          f"HBM = {HBM_TBPS} TB/s)")
    print(f"{'='*78}")
    keys = ["label", "N", "dtype", "GFLOPS", "GBps",
            "intensity_FLOP_per_Byte", "roofline_predicted_GFLOPS",
            "efficiency_wrt_roofline", "HBM_util_pct", "Vector_peak_util_pct"]
    widths = [28, 10, 8, 10, 10, 18, 22, 20, 14, 20]
    def _row(vals):
        return " ".join(f"{str(v)[:w]:>{w}}" for v, w in zip(vals, widths))
    print(_row(keys))
    print("-" * sum(widths))
    for p in points:
        print(_row([p.get(k, "—") for k in keys]))


# =============================================================================
# 8. main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="跑四家 (NumPy/ TorchNPU-native / Triton-Ascend / TileLang-Ascend / "
                    "Ascend-C scalar+production) GELU 微基准 + Roofline 摘要.\n"
                    "提示: 若只用某张 NPU, 先 export ASCEND_RT_VISIBLE_DEVICES=<phy_id> 再传 "
                    "--device 0 更稳.")
    ap.add_argument("--sizes",
                    default="65536,524288,1048576,8388608,33554432,67108864,134217728",
                    help="逗号分隔的 7 档 N (64K ~ 128M)")
    ap.add_argument("--repeats", type=int, default=15,
                    help="每档 N 重复次数, 取 best ms (默认 15)")
    ap.add_argument("--device", type=int, default=0,
                    help="NPU 逻辑设备号 (ASCEND_RT_VISIBLE_DEVICES 后, 默认 0)")
    ap.add_argument("--block-size", type=int, default=1024,
                    help="Triton BLOCK_SIZE (默认 1024)")
    ap.add_argument("--which", default="both",
                    choices=["scalar", "prod", "both"],
                    help="[仅 ascendc 分支] 跑 scalar 地板版 / prod 生产版 / both 都跑 (默认 both)")
    ap.add_argument("--run", default="all",
                    help="csv of tags: numpy,torch,triton,tilelang,ascendc (默认 all)")
    ap.add_argument("--out", default="",
                    help="写出完整 JSON (例如 bench_gelu_full.json)")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x]
    tags = set(args.run.split(","))
    run_all = "all" in tags

    result = dict(
        SoC="Ascend 910B2",
        CANN_version="9.0.0",
        device_id_logical=args.device,
        ascendc_which=args.which,
        repeats_per_size=args.repeats,
        sizes=sizes,
        THEORETICAL_PEAK_TFLOPS_FP16_VECTOR=THEORETICAL_PEAK_TFLOPS_FP16_VECTOR,
        HBM_TBPS_QUOTED=HBM_TBPS,
        FLOPS_PER_ELEMENT=FLOPS_PER_ELEMENT,
        FLOPs_Intensity_note="FLOPs_per_elem * N / (2 * elem_bytes * N) → "
                             "fp16 theoretical I ≈ 11/4 = 2.75 FLOP/Byte (memory-bound)",
    )

    # ---- 各分支按序跑 ----
    numpy_rows = None
    torch_rows = None
    triton_rows = None
    tile_rows = None
    ascendc_rows = None

    if run_all or "numpy" in tags:
        numpy_rows = numpy_bench(sizes, args.repeats, dtype=np.float32)
        result["numpy_cpu_fp32"] = numpy_rows
        _print_table("1/5 · NumPy CPU (fp32 参考基线)", numpy_rows)

    if run_all or "torch" in tags:
        try:
            torch_rows = torch_npu_gelu_bench(sizes, args.repeats, args.device, dtype="fp16")
        except Exception as e:
            torch_rows = [dict(error=f"torch_npu_bench exc {type(e).__name__}: {e}")]
        result["torch_npu_fp16"] = torch_rows
        _print_table("2/5 · PyTorch NPU F.gelu (TBE/TIK 原生 fp16)", torch_rows)

    if run_all or "triton" in tags:
        try:
            triton_rows = triton_bench(sizes, args.repeats, args.device,
                                       dtype="fp16", block_size=args.block_size)
        except Exception as e:
            triton_rows = [dict(error=f"triton_bench exc {type(e).__name__}: {e}")]
        result["triton_npu_fp16"] = triton_rows
        _print_table("3/5 · Triton-Ascend GELU (fp16, BLOCK=%d)" % args.block_size, triton_rows)

    if run_all or "tilelang" in tags:
        try:
            tile_rows = tilelang_bench(sizes, args.repeats)
        except Exception as e:
            _annotate_tl_error(e)
            print(f"[SKIP] tilelang_bench failed: {type(e).__name__}: {e}{_TL_ERR_HINT}")
            tile_rows = None
        result["tilelang_npu_fp16"] = tile_rows
        _print_table("4/5 · TileLang-Ascend GELU (fp16)", tile_rows or [{"error": "skipped"}])

    if run_all or "ascendc" in tags:
        try:
            ascendc_rows = ascend_c_bench(sizes, args.repeats, which=args.which)
        except Exception as e:
            ascendc_rows = {"ascendc": [{"error": f"ascend_c_bench exc {type(e).__name__}: {e}"}]}
        result["ascendc_npu_fp16"] = ascendc_rows  # dict: sublabel → rows
        for sub, rows in (ascendc_rows or {}).items():
            _print_table(f"5/5 · Ascend-C GELU fp16 [{sub}]", rows)

    # ---- Roofline ----
    rl = roofline_summary(
        NumPy_ref_CPU=numpy_rows or [],
        TorchNPU_Native_TIK_TBE_910B2=torch_rows or [],
        Triton_Ascend_910B2=triton_rows or [],
        TileLang_Ascend_910B2=tile_rows or [],
        AscendC_910B2=ascendc_rows or {},
    )
    result["roofline_points"] = rl
    _print_roofline(rl)

    # ---- JSON ----
    txt = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"\n[INFO] 完整 JSON 已写入: {Path(args.out).resolve()}")
    else:
        print("\n===== JSON OUTPUT =====")
        print(txt)


if __name__ == "__main__":
    main()
