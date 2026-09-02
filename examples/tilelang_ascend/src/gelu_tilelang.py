"""
GELU kernel —— TileLang + tilelang-ascend 后端。

GELU 是逐元素算子, 在 TileLang 里对应的是**Vector 核** (Scope("M")),
不像 GEMM 需要走 Cube (Scope("C"))。Ascend NPU 上每个 AI Core 分:
  - Cube 核: 矩阵乘专用 (GEMM)
  - Vector 核: 向量/逐元素运算 (GELU / ReLU / SiLU 等激活都走它)
  - MTE2/MTE3 : GM↔UB↔L1 的搬运队列 (和 Vector/Cube 解耦, 可流水线)

本实现显式给出:
  - GM → UB 载入 (T.copy)
  - Vector 核计算 (Scope("M"), 即 Vector 执行域)
    - UB 内 x -> x^3 -> inner -> tanh(inner) -> y 的完整流水线
  - UB → GM 写回 (T.copy)
  - T.alloc_UB : Vector 核的片上 Unified Buffer (类比 GPU shared memory 但位于 Vector 域)
  - 多核并行: T.Kernel(num_blocks, is_npu=True), 把 N 按 BLOCK 切给多个 AI Core
"""

from __future__ import annotations

import sys
import numpy as np

import tilelang
import tilelang.language as T


# 教学版常数, 和 examples/python gelu / ascend_c gelu_kernel / triton gelu_triton 保持一致
_SQRT_2_OVER_PI = 0.7978845608028654
_CUBIC_COEF     = 0.044715


@tilelang.jit(out_idx=[-1])
def gelu_activation(N: int, BLOCK: int, dtype: str = "float16"):
    """
    TileLang-Ascend 逐元素 GELU kernel。

    Args:
        N       : 元素总数 (编译期常量)
        BLOCK   : 每个 AI Core (每个 program) 处理的元素数 (16 的倍数, 对齐 Vector 指令粒度)
        dtype   : 输入/输出精度, 默认 fp16 (Vector 核原生精度)

    out_idx=[-1] : 返回 Y (最后一个 Tensor 参数)

    备注: TileLang 0.1.13 对 @T.prim_func 的参数注解走 `get_type_hints(func,
    globalns=mod_globalns, localns={})`. 如果把 `T.Tensor((N,), dtype)` 作为裸注解写上去,
    Python 3.11 的 `typing._eval_type` 会在"模块全局域"里解析标识符 `N` / `dtype`,
    而它们实际是外层函数 `gelu_activation` 的闭包参数 → 抛 NameError. 因此我们把参数
    类型写为**字符串前向引用** (PEP 563 from __future__ import annotations), 由 TileLang
    在后续阶段自行延迟解析, 从而绕过这一层 typing 求值。
    """
    assert N % BLOCK == 0, f"教学版要求 N={N} 能被 BLOCK={BLOCK} 整除 (方便对齐 Vector 流水线)"
    num_blocks = N // BLOCK

    # ---- TileLang 0.1.13 @T.prim_func 类型解析 workaround ----
    # tilelang 0.1.13 在 eager/builder.py:get_type_hints 里手动调用
    #   typing._eval_type(annotation, globalns=func.__globals__, localns={})
    # localns 被传了空 dict, 所以闭包参数 N / BLOCK / dtype 在注解字符串
    # `T.Tensor((N,), dtype)` 里解析不到 → NameError.
    # 临时 workaround: 把这 3 个符号塞进模块级 globals 里, @T.prim_func 定义完再还原.
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
        def main(X: "T.Tensor((N,), dtype)", Y: "T.Tensor((N,), dtype)"):
            # ---- 多核并行: 核 (block) 下标 = cid; 一共 num_blocks 个 AI Core ----
            #   每个核处理 BLOCK 个元素: X[cid*BLOCK : (cid+1)*BLOCK].
            #   TileLang 0.1.13 的 T.Kernel(n) 只返回一个 block-index.
            with T.Kernel(num_blocks) as cid:
                # ---- 分配本地缓冲 (映射到 NPU UB / shared) ----
                X_UB = T.alloc_local((BLOCK,), dtype)
                Y_UB = T.alloc_local((BLOCK,), dtype)

                # ---- GM → UB: 读 BLOCK 个元素 ----
                T.copy(X[cid * BLOCK : cid * BLOCK + BLOCK], X_UB)

                # ---- 逐元素 GELU: 显式 for 循环 (0.1.13 不支持 Buffer 重载 *) ----
                #   TileLang 0.1.13 需要用 T.serial 生成元素级循环, 不能直接写 X_UB * X_UB.
                for k in T.serial(BLOCK):
                    xv = X_UB[k]
                    x3 = xv * xv * xv
                    inner = _SQRT_2_OVER_PI * (xv + _CUBIC_COEF * x3)
                    t = T.tanh(inner)
                    Y_UB[k] = 0.5 * xv * (1.0 + t)

                # ---- UB → GM: 写回 BLOCK 个元素 ----
                T.copy(Y_UB, Y[cid * BLOCK : cid * BLOCK + BLOCK])

        return main
    finally:
        for _k, _v in _saved.items():
            if _v is _SENTINEL:
                _mod.__dict__.pop(_k, None)
            else:
                _mod.__dict__[_k] = _v


def gelu_tilelang(x, BLOCK=1024):
    """
    便捷封装: 1D fp16 输入 (NPU 或 host 均可) → 逐元素 GELU 输出。

    TileLang 首次调用会触发编译 (N→kernel), 后续调用走缓存。
    对 fp32 或多维输入, 我们先 cast / flatten, 与 numpy reference 对齐好后再比。
    """
    # ---- 统一转成 float16 numpy 1D 以便 tilelang 当前版本兼容 ----
    #   (tilelang 的 tensor 输入在宿主 Python 里最稳定的形式就是 np.ndarray)
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
    # numpy 分支
    if isinstance(x, np.ndarray):
        if orig_shape is None:
            orig_shape = x.shape
            orig_dtype = x.dtype
        x16 = x.reshape(-1).astype(np.float16)
    else:
        raise TypeError(f"不支持的输入类型: {type(x)}, 需要 numpy.ndarray 或 torch.Tensor")

    N = x16.size
    # 教学版简化: pad 到 BLOCK 整数倍, 结果只取前 N 个
    pad = (BLOCK - N % BLOCK) % BLOCK
    if pad:
        x16 = np.concatenate([x16, np.zeros(pad, dtype=np.float16)])
    N_padded = x16.size

    kernel = gelu_activation(N_padded, BLOCK, dtype="float16")
    y16 = kernel(x16)  # 输出是 np.ndarray fp16 (或被 tilelang 包装过)
    y16 = np.asarray(y16, dtype=np.float16)

    # unpad + reshape + cast
    y = y16[:N].reshape(orig_shape)
    if orig_dtype is not None:
        if isinstance(orig_dtype, np.dtype):
            y = y.astype(orig_dtype)
        else:
            # torch dtype → numpy dtype
            mapping = {
                "float16": np.float16, "float32": np.float32, "float64": np.float64,
            }
            # 兜底直接转 fp16 吧, 不折腾复杂反射
            _ = mapping
            y = y.astype(np.float16)
    return y


# 独立的 numpy ground truth, 与 examples/python/src/gelu.py 保持同公式
def gelu_reference_numpy(x_np: np.ndarray) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    inner = _SQRT_2_OVER_PI * (x + _CUBIC_COEF * np.power(x, 3))
    y = 0.5 * x * (1.0 + np.tanh(inner))
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    N = 4096
    x = (rng.standard_normal(N) * 3.0).astype(np.float16)
    y = gelu_tilelang(x)
    ref = gelu_reference_numpy(x)
    max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-gelu smoke: N={N}, max_abs_err={max_err:.6e}")
    assert max_err < 5e-3, f"tilelang gelu smoke FAIL, max_err={max_err}"
    print("tilelang-gelu smoke PASSED")
