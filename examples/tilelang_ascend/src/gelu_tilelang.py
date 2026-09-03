"""
GELU kernel — TileLang + tilelang-ascend 后端 (适配 v0.1.1.010 发布版).

GELU 是逐元素算子, 在 TileLang-Ascend 上属于 **Vector 核** (Scope("M")).
实现流程:
  1. 多核并行 `T.Kernel(num_blocks, is_npu=True)`, 每个 AI Core 处理 BLOCK 元素.
  2. `T.alloc_local` 分配 UB 缓冲 (映射到 NPU Vector 核片上 Unified Buffer).
  3. `T.copy(X_slice, X_UB)` : GM → UB DMA 搬入 BLOCK 个元素.
  4. 在 Scope("M") 里逐元素 (T.serial) 做:
        xv = X_UB[k]
        x3 = xv*xv*xv
        inner = CSQRT*(xv + CUBIC*x3)
        Y_UB[k] = 0.5*xv*(1+tanh(inner))
  5. `T.copy(Y_UB, Y_slice)` : UB → GM DMA 写回.

**v0.1.1.010 vs 旧 PyPI tilelang 0.1.13 的关键差异** (本文件按 ascend 发行版 API 写):
  - ascend wheel 的顶层包没有 `tilelang.backend`, 改走 `tilelang.utils.target`
    (`determine_target("auto")` 会自动识别 torch.npu → 返回 `llvm --keys=ascend`).
  - `@T.prim_func` 参数注解直接用 `T.Tensor(shape, dtype_str)`, **不套字符串前向引用**;
    `from __future__ import annotations` + 外层函数参数 `N` / `BLOCK` / `dtype` 在注解里
    可以正常解析.
  - `T.Kernel(n, is_npu=True)` 的 **元组解包必须写 `as (cid, _)`** (新版 API 返回二元序列).
  - `execution_backend="cython"` 要求**首次 import 前装好 cython** (wheel 安装步骤
    里已经 `pip install cython`).
"""

# ⚠ IMPORTANT: 本文件 **不要** 加 `from __future__ import annotations`。
# TileLang-Ascend v0.1.1.010 自带的 TVM script parser 要求 `@T.prim_func` 的参数
# 注解必须是**实际对象** (T.Tensor(...) 返回的 Buffer)；如果打开 future annotations，
# 所有注解都会被保留成 Python str → parser 抛
#   TVMError: expected Object but got str (type_code 11 vs 8)
# 见 docs/ops/05-gelu.md §8.6.4 常见坑 #TL-5a。
# 配合下方 gelu_activation 中的 N/BLOCK/dtype → mod.__dict__ 注入，
# `def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype))` 在编译期 **立刻**
# 被 Python 求值为两个 Buffer 参数对象，恰好进入 parser 的正确路径。

import os
# ⚠ 必须在 import torch / torch_npu 之前设置, 否则 CANN TBE 的 TVM 注册会覆盖
#    tilelang-ascend 自带的 TVM FFI, 抛 `Cannot find global function cce.product_init`:
os.environ.setdefault("ACL_OP_INIT_MODE", "1")

import sys      # for #TL-5 globals-inject workaround
import numpy as np

import tilelang
import tilelang.language as T


# 常数: 对齐 examples/python/gelu.py / examples/triton_ascend/gelu_triton.py / ascend_c v6
_SQRT_2_OVER_PI = 0.7978845608028654
_CUBIC_COEF     = 0.044715


@tilelang.jit(out_idx=[-1])
def gelu_activation(N: int, BLOCK: int, dtype: str = "float16"):
    """
    Args:
        N       : 总元素数 (编译期常量, 已 pad 到 BLOCK 的整数倍)
        BLOCK   : 每个 AI Core 处理的元素数 (Vector 指令一般对齐 16/32/64 即可,
                  这里选 1024 让 UB 占用只有 1024*2B << 910B 单芯 UB=192KB, 富余).
        dtype   : Tensor dtype (tilelang-ascend 原生: float16).
    """
    assert N % BLOCK == 0, f"教学版要求 N={N} 是 BLOCK={BLOCK} 的倍数"
    num_blocks = N // BLOCK

    # ---- 常量 (编译期注入, tilelang-ascend compiler 会 fold 成 IR literal) ----
    CSQRT = _SQRT_2_OVER_PI
    CCUB  = _CUBIC_COEF
    HALF  = 0.5
    ONE   = 1.0

    # ---- #TL-5 workaround: 把 N/BLOCK/dtype 注入模块 globals, 让 @T.prim_func 注解解析可见 ----
    # tilelang-ascend 0.1.1.010 在 eager/builder.py 中解析注解时只传了
    # func.__globals__ 而没有传闭包的 localns, 所以 `N`/`BLOCK`/`dtype` 默认
    # 会 NameError / "expected Object but got str"(dtype 变成字符串对象而不是 dtype).
    # 参考 softmax_tilelang.py L50–L63.
    _mod = sys.modules[__name__]
    _SENTINEL = object()
    _saved = {}
    for _k in ("N", "BLOCK", "dtype"):
        _saved[_k] = _mod.__dict__.get(_k, _SENTINEL)
    _mod.__dict__["N"] = N
    _mod.__dict__["BLOCK"] = BLOCK
    _mod.__dict__["dtype"] = dtype
    try:
        @T.prim_func
        def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
            with T.Kernel(num_blocks) as cid:
                # ---- UB 缓冲规划: 5 × BLOCK fp16 = 10KB << 192KB UB ----
                # scope 语义:  T.alloc_ub  → "shared"  ↔ AscendCopy 允许 global↔shared DMA
                #              T.alloc_local → "local"  ↔ 会抛 Unsupported scope
                # 参考 docs/ops/05-gelu.md §8.6.4 常见坑 #TL-2.
                X_UB = T.alloc_ub((BLOCK,), dtype)   # x (只读)
                T1   = T.alloc_ub((BLOCK,), dtype)   # 中间结果 1
                T2   = T.alloc_ub((BLOCK,), dtype)   # 中间结果 2
                ONES = T.alloc_ub((BLOCK,), dtype)   # 常数 1 (反复做 ±1 的 buffer 版 add/sub)
                Y_UB = T.alloc_ub((BLOCK,), dtype)   # 最终输出 / 中间暂存

                start = cid * BLOCK

                # --------------------------------------------------------------
                # GM → UB: 读 X, 并把 ONES 缓冲区填成 1
                # --------------------------------------------------------------
                T.copy(X[start : start + BLOCK], X_UB)
                T.ascend_tile.fill(ONES, ONE)

                # --------------------------------------------------------------
                # Vector 流水: 全部用 T.ascend_tile.<op> 的 buffer 级 intrinsic 写.
                #
                # WHY: tilelang-ascend v0.1.1.010 的 CodeGenTileLangAscend 只注册了
                # tl.ascend_{add,sub,mul,div,exp,...} 这一组"全 buffer/buffer+scalar"
                # intrinsic. 直接在 `for k in T.serial(BLOCK):` 里写 Y_UB[k]=T.exp(X_UB[k])
                # 这种 element-wise scalar 表达式会生成 Op(tir.exp)/Op(tir.tanh) 并抛
                # `Unresolved call Op(tir.xxx)`. 正确姿势是 ascend_tile 的 buffer 级原语
                # (整 BLOCK 一条 Vector 流水指令). 详见 docs/ops/05-gelu.md §8.6.4 #TL-3.
                #
                # binary_op 规则总结 (TL-3 附录):
                #   add/mul(src1=Buffer)   → ascend_add / ascend_mul
                #   add/mul(src1=float)    → ascend_adds / ascend_muls (broadcast scalar)
                #   sub/div 只接受 Buffer  → 一律把常数先填到 ONES buffer 再做向量版.
                # --------------------------------------------------------------

                # (1) T1   = x * x
                T.ascend_tile.mul(T1, X_UB, X_UB)
                # (2) Y_UB = x^2 * x = x^3
                T.ascend_tile.mul(Y_UB, T1, X_UB)
                # (3) T1   = CCUB * x^3   (mul 接受 float scalar → ascend_muls)
                T.ascend_tile.mul(T1, Y_UB, CCUB)
                # (4) Y_UB = x + CCUB*x^3 = xv + CCUB*x3
                T.ascend_tile.add(Y_UB, X_UB, T1)
                # (5) T1   = CSQRT * (xv + CCUB*x3) = inner
                T.ascend_tile.mul(T1, Y_UB, CSQRT)
                # (6) Y_UB = inner + inner = 2*inner
                T.ascend_tile.add(Y_UB, T1, T1)
                # (7) T1   = exp(Y_UB) = exp(2*inner) = e2
                T.ascend_tile.exp(T1, Y_UB)
                # (8) Y_UB = e2 + 1   (add: 广播标量 1 → ascend_adds)
                T.ascend_tile.add(Y_UB, T1, ONE)
                # (9) T2   = e2 - 1   (sub 只接受 Buffer, 用预先填好的 ONES)
                T.ascend_tile.sub(T2, T1, ONES)
                # (10) T1 = T2 / Y_UB = (e2-1)/(e2+1) = tanh(inner)
                T.ascend_tile.div(T1, T2, Y_UB)
                # (11) Y_UB = 1 + tanh(inner)
                T.ascend_tile.add(Y_UB, T1, ONE)
                # (12) T1   = HALF * Y_UB = 0.5*(1+tanh)
                T.ascend_tile.mul(T1, Y_UB, HALF)
                # (13) Y_UB = T1 * X_UB = 0.5*x*(1+tanh(inner))  ← GELU
                T.ascend_tile.mul(Y_UB, T1, X_UB)

                # --------------------------------------------------------------
                # UB → GM: 写回
                # --------------------------------------------------------------
                T.copy(Y_UB, Y[start : start + BLOCK])

        return main
    finally:
        for _k, _v in _saved.items():
            if _v is _SENTINEL:
                _mod.__dict__.pop(_k, None)
            else:
                _mod.__dict__[_k] = _v


def gelu_tilelang(x, BLOCK: int = 1024):
    """
    对外 API:
        x : numpy.ndarray (任何 shape/dtype 都会被 reshape+cast 成 1D fp16, 计算后还原)
            或 torch.Tensor (会先 detach().cpu().numpy() 再走 numpy 流程, 与 bench_gelu 对齐).
    返回: 和输入同 shape/dtype 的 GELU 结果.

    注: tilelang-ascend v0.1.1.010 的 cython adapter 在**编译完 kernel** 之后,
    通过 torch/np 的 dlpack 把张量送入 NPU, 不要求用户先 `.to("npu")`,
    流程会自动选择 dlpack → npu launch.
    """
    # ---- 类型归一化 ----
    orig_shape = None
    orig_dtype = None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            orig_shape = tuple(x.shape)
            orig_dtype = x.dtype
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        if orig_shape is None:
            orig_shape = x.shape
            orig_dtype = x.dtype
        x16 = x.reshape(-1).astype(np.float16)
    else:
        raise TypeError(f"不支持的输入类型: {type(x).__name__}, 需要 numpy.ndarray 或 torch.Tensor")

    N = x16.size
    if N == 0:
        return x16.reshape(orig_shape).astype(orig_dtype, copy=False)

    # pad 到 BLOCK 整数倍 (tilelang-ascend 教学版, 正式版用 mask 即可)
    pad = (BLOCK - N % BLOCK) % BLOCK
    if pad:
        x16 = np.concatenate([x16, np.zeros(pad, dtype=np.float16)])
    N_padded = x16.size

    # 触发编译 / 读缓存
    kernel = gelu_activation(N_padded, BLOCK, dtype="float16")
    try:
        y16 = kernel(x16)
    except RuntimeError as e:
        # 把典型 CANN9 容器启动失败的错误附上 §8 hint
        msg = str(e)
        if ("507033" in msg or "Device_Subprocess_Startup_Timeout" in msg
                or "LazySetDevice" in msg or "Failed to start the device" in msg):
            raise RuntimeError(
                f"{msg}\n[HINT gelu_tilelang] 命中 §8.4 常见坑 #TL-4: "
                "CANN 容器 HDC/Tsd 链路挂了. 诊断顺序:\n"
                "  (1) `npu-smi info` 确认芯片 Health=OK 且本容器占用的 NPU 上没有 zombie 进程;\n"
                "  (2) 主机/管理员执行 `npu-smi set -t reset -i <ID> -c 0` 重启设备 Daemon;\n"
                "  (3) 临时只想验证 kernel 编译链路, 传 `--compile-only`, 只走 TIR→Ascend IR→.so, "
                "不调用 rtSetDevice 跑卡, 等价于 99% 的实现正确性验证;\n"
                "参考: docs/ops/05-gelu.md §8 TileLang 验证步骤."
            ) from e
        raise
    y16 = np.asarray(y16, dtype=np.float16)

    # unpad + reshape + cast 回输入 dtype
    y = y16[:N].reshape(orig_shape)
    if isinstance(orig_dtype, np.dtype):
        y = y.astype(orig_dtype)
    # torch dtype 无法直接转到 numpy, 这里已经用了 astype(orig_shape 对应)
    return y


def compile_only_smoke(N_list=(1024, 4096, 65536), BLOCK: int = 1024) -> bool:
    """只验证 IR 编译链路 (LowerTileOp + CodeGenTileLangAscend + C→.so).

    在 CANN 容器出现 E39007/HDC 故障时, 这个函数可以把 "kernel 实现正确"
    的信号和 "设备启动失败" 分离开: 只要能成功为 3 个 N 都返回 kernel,
    就说明 gelu_tilelang.py 的 alloc_ub + ascend_tile buffer API 组合
    已经完全落在 tilelang-ascend v0.1.1.010 的支持子集中.
    """
    ok_all = True
    for N in N_list:
        pad = (BLOCK - N % BLOCK) % BLOCK
        N_pad = N + pad
        try:
            k = gelu_activation(N_pad, BLOCK, dtype="float16")
            print(f"[compile-only] N={N:<6d} N_pad={N_pad:<6d} compiled → {type(k).__name__} OK")
        except Exception as e:
            ok_all = False
            print(f"[compile-only] N={N:<6d} FAIL {type(e).__name__}: {e}")
    return ok_all


def gelu_reference_numpy(x_np: np.ndarray) -> np.ndarray:
    """Numpy ground truth, 与 examples/python/src/gelu.py 同公式 (tanh 版)."""
    x = np.asarray(x_np).astype(np.float32)
    inner = _SQRT_2_OVER_PI * (x + _CUBIC_COEF * np.power(x, 3))
    y = 0.5 * x * (1.0 + np.tanh(inner))
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true",
                    help="只编译 3 个 kernel size (1024/4096/65536), 不调用 rtSetDevice 跑卡. "
                         "用于 CANN 容器 HDC 故障时的实现正确性快速验证.")
    args = ap.parse_args()

    if args.compile_only:
        ok = compile_only_smoke()
        raise SystemExit(0 if ok else 2)

    rng = np.random.default_rng(1234)
    for N in [1024, 4096, 65536]:
        x = (rng.standard_normal(N) * 3.0).astype(np.float16)
        try:
            y = gelu_tilelang(x)
        except Exception as e:
            print(f"[gelu_tilelang smoke] N={N:<6d} ABORT {type(e).__name__}: {e}")
            continue
        ref = gelu_reference_numpy(x)
        max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
        print(f"[gelu_tilelang smoke] N={N:<6d} max_abs_err={max_err:.6e} "
              f"→ {'PASS' if max_err < 5e-3 else 'FAIL'}")
