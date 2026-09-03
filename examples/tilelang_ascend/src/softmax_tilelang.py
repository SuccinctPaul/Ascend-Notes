"""
Softmax kernel —— TileLang + tilelang-ascend 后端。

与 GELU 不同, Softmax 不是逐元素算子, 它带有沿最后一维的 REDUCTION:
    y[i] = exp(x[i] - m) / Σ_j exp(x[j] - m),  其中 m = max_j x[j]

因为 TileLang 0.1.13 没有显式 ReduceMax / ReduceSum 原语 (或者还需进一步配置),
本教学版在 UB 内通过 **T.serial 循环** 手工完成 4 个阶段:
    Phase 1 : 串行比较, 找出整行最大值 m
    Phase 2 : 逐元素 exp(x[i] - m)
    Phase 3 : 串行累加, 求出分母 sum_exp
    Phase 4 : 逐元素 exp_val / sum_exp (广播除法)

每个 AI Core 负责完整的一行特征, 因此输入形状只支持 (D,) 的 1D kernel;
若用户传 (M, D) 2D, 封装器 `softmax_tilelang` 对每行分别调用 1D kernel。
"""

from __future__ import annotations

import sys
import numpy as np

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def softmax_1d(D: int, BLOCK: int, dtype: str = "float16"):
    """
    TileLang-Ascend 1D Softmax kernel (沿长度 D 做归约)。

    Args:
        D       : 向量元素总数 (编译期常量, 教学版要求 D % BLOCK == 0 且 BLOCK == D)
        BLOCK   : UB 缓冲大小, 应为 D 的整倍数; 教学版取 BLOCK == D (一整行驻留 UB)
        dtype   : 输入/输出精度, 默认 fp16 (Vector 核原生精度)

    out_idx=[-1] : 返回 Y (最后一个 Tensor 参数)

    备注: TileLang 0.1.13 对 @T.prim_func 的参数注解走 `get_type_hints(func,
    globalns=mod_globalns, localns={})`. 如果把 `T.Tensor((D,), dtype)` 作为裸注解写上去,
    Python 3.11 的 `typing._eval_type` 会在"模块全局域"里解析标识符 `D` / `dtype`,
    而它们实际是外层函数 `softmax_1d` 的闭包参数 → 抛 NameError. 因此我们把参数
    类型写为**字符串前向引用** (PEP 563 from __future__ import annotations), 由 TileLang
    在后续阶段自行延迟解析, 从而绕过这一层 typing 求值。
    """
    assert D % BLOCK == 0, f"教学版要求 D={D} 能被 BLOCK={BLOCK} 整除 (一整行驻留 UB)"
    # 教学版简化: 每个 AI Core 处理完整的一行 → num_blocks = 1
    num_blocks = D // BLOCK

    # ---- TileLang 0.1.13 @T.prim_func 类型解析 workaround ----
    # tilelang 0.1.13 在 eager/builder.py:get_type_hints 里手动调用
    #   typing._eval_type(annotation, globalns=func.__globals__, localns={})
    # localns 被传了空 dict, 所以闭包参数 D / BLOCK / dtype 在注解字符串
    # `T.Tensor((D,), dtype)` 里解析不到 → NameError.
    # 临时 workaround: 把这 3 个符号塞进模块级 globals 里, @T.prim_func 定义完再还原.
    _mod = sys.modules[__name__]
    _SENTINEL = object()
    _saved = {}
    for _k in ("D", "BLOCK", "dtype"):
        _saved[_k] = _mod.__dict__.get(_k, _SENTINEL)
    _mod.__dict__["D"] = D
    _mod.__dict__["BLOCK"] = BLOCK
    _mod.__dict__["dtype"] = dtype
    try:
        @T.prim_func
        def main(X: "T.Tensor((D,), dtype)", Y: "T.Tensor((D,), dtype)"):
            # ---- 多核并行: 核 (block) 下标 = cid; 教学版一行一个 AI Core ----
            with T.Kernel(num_blocks) as cid:
                # ---- 分配本地缓冲 (映射到 NPU UB / shared) ----
                # X_UB/Y_UB : 一整行数据, 长度 BLOCK
                X_UB = T.alloc_local((BLOCK,), dtype)
                Y_UB = T.alloc_local((BLOCK,), dtype)
                # 标量缓冲: 最大值 / exp 和 / 倒数
                M_UB = T.alloc_local((1,), dtype)   # 整行最大值 m
                S_UB = T.alloc_local((1,), dtype)   # Σ exp(x - m)
                INV_UB = T.alloc_local((1,), dtype) # 1 / S_UB

                start = cid * BLOCK

                # ---- GM → UB: 读 BLOCK 个元素 ----
                T.copy(X[start : start + BLOCK], X_UB)

                # ---- Phase 1: 串行求整行最大值 ----
                M_UB[0] = X_UB[0]
                for k in T.serial(1, BLOCK):
                    xv = X_UB[k]
                    if xv > M_UB[0]:
                        M_UB[0] = xv

                # ---- Phase 2: 逐元素 exp(x - m), 写入 Y_UB 暂存 ----
                for k in T.serial(BLOCK):
                    diff = X_UB[k] - M_UB[0]
                    Y_UB[k] = T.exp(diff)

                # ---- Phase 3: 串行求和 S = Σ Y_UB[k] ----
                S_UB[0] = Y_UB[0]
                for k in T.serial(1, BLOCK):
                    S_UB[0] = S_UB[0] + Y_UB[k]

                # ---- Phase 4: inv = 1/S, 逐元素 Y = exp_val * inv ----
                #   除法实现为 1.0/S 再乘以每个 Y_UB[k] (Vector 核 native 支持 mul)
                INV_UB[0] = 1.0 / S_UB[0]
                for k in T.serial(BLOCK):
                    Y_UB[k] = Y_UB[k] * INV_UB[0]

                # ---- UB → GM: 写回 BLOCK 个元素 ----
                T.copy(Y_UB, Y[start : start + BLOCK])

        return main
    finally:
        for _k, _v in _saved.items():
            if _v is _SENTINEL:
                _mod.__dict__.pop(_k, None)
            else:
                _mod.__dict__[_k] = _v


def softmax_tilelang(x, BLOCK=256):
    """
    便捷封装: 1D / 2D fp16/fp32 输入 (NPU 或 host 均可) → 沿最后一维 softmax 输出。

    - 对 1D (D,): 直接调用一次 softmax_1d kernel
    - 对 2D (M, D): 对 M 行每行调用一次 1D kernel, 拼接结果
    - 对更高维: 先 reshape 成 2D (..., D), 处理后再还原 shape

    TileLang 首次调用会触发编译 (D→kernel), 后续相同 D 走缓存。
    """
    # ---- 统一转成 numpy, 保存原始 shape/dtype ----
    orig_shape = None
    orig_dtype = None
    orig_ndim = None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            orig_shape = tuple(x.shape)
            orig_dtype = x.dtype
            orig_ndim = len(orig_shape)
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    # numpy 分支
    if isinstance(x, np.ndarray):
        if orig_shape is None:
            orig_shape = x.shape
            orig_dtype = x.dtype
            orig_ndim = len(orig_shape)
        x = np.asarray(x)
    else:
        raise TypeError(f"不支持的输入类型: {type(x)}, 需要 numpy.ndarray 或 torch.Tensor")

    # 把更高维展平到 2D: 前 N-1 维合并成 batch, 最后一维是特征 D
    D = orig_shape[-1]
    batch_shape = orig_shape[:-1]
    if batch_shape:
        M = int(np.prod(batch_shape))
    else:
        M = 1
    x2d = x.reshape(M, D)

    # 教学版默认 fp16 进 kernel (和 GELU 保持一致的精度策略)
    x16 = x2d.astype(np.float16)

    # pad D 到 BLOCK 整数倍: 每个 kernel 调用一次整行
    pad = (BLOCK - D % BLOCK) % BLOCK
    D_padded = D + pad
    if pad:
        x16 = np.pad(x16, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    # ---- 对每行调用 1D kernel ----
    # kernel 编译一次后缓存 (相同 D_padded + BLOCK 复用)
    kernel = softmax_1d(D_padded, D_padded, dtype="float16")  # BLOCK == D_padded

    y_rows = []
    for i in range(M):
        row_in = x16[i]  # (D_padded,)
        row_out = kernel(row_in)
        y_rows.append(np.asarray(row_out, dtype=np.float16))

    y16_2d = np.stack(y_rows, axis=0)  # (M, D_padded)

    # unpad
    if pad:
        y16_2d = y16_2d[:, :D]

    # reshape 回原始 shape
    y = y16_2d.reshape(orig_shape)

    # cast 回原始 dtype
    if orig_dtype is not None:
        if isinstance(orig_dtype, np.dtype):
            y = y.astype(orig_dtype)
        else:
            # torch dtype → numpy dtype; 兜底走 fp16
            mapping = {
                "float16": np.float16, "float32": np.float32, "float64": np.float64,
            }
            _ = mapping
            y = y.astype(np.float16)
    return y


# 独立的 numpy ground truth, 与 examples/python/src/softmax.py 保持同公式
def softmax_reference_numpy(x_np: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    y = e / s
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    D = 512
    x = (rng.standard_normal(D) * 3.0).astype(np.float16)
    y = softmax_tilelang(x)
    ref = softmax_reference_numpy(x)
    max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-softmax smoke: D={D}, max_abs_err={max_err:.6e}")
    # softmax fp16 归约累加会放大误差, 这里给 1e-2 容差
    assert max_err < 1e-2, f"tilelang softmax smoke FAIL, max_err={max_err}"
    print("tilelang-softmax smoke PASSED")
