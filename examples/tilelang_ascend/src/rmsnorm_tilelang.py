"""
RMSNorm kernel —— TileLang + tilelang-ascend 后端。

    rms  = sqrt( (1/D) · Σ_j x_j² + eps )
    y[m,:] = (X[m,:] / rms) · gamma[:]

实现走 tilelang-ascend 官方的 **Vector tile 内建指令** (T.tile.*) 风格
(参考官方 examples/normalization/rms_norm.py 与 examples/elementwise/
elementwise_add.py), 每条指令与 docs/02 §5.2 的 Vector 指令表一一对应:

    T.tile.cast      : fp16 → fp32 (Vector 主精度 fp16, 归约用宽精度)
    T.tile.mul       : 逐元素平方
    T.reduce_sum     : 沿特征维归约 Σx²
    T.tile.rsqrt     : 1/sqrt(Σx²/D + eps) —— "乘倒数" 而非逐元素除
    T.tile.broadcast : 把标量 inv_rms 广播回整行
    T.tile.mul ×2    : y = x · inv_rms · gamma

kernel 接收 **torch NPU 张量** (tilelang-ascend 的 cython adapter 只认
torch tensor), 2D 一次 launch 处理所有行; 物理核按 (cid, vid) 双 Vector
核拆分, 每核处理 2 行 (M 为偶数, wrapper 负责补齐)。
"""

import numpy as np

import tilelang
import tilelang.language as T

# 官方推荐 pass 配置 (自动 Cube/Vector 同步; 本 kernel 纯 Vector)
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def rmsnorm_2d(M: int, D: int, dtype: str = "float16", eps: float = 1e-6):
    """
    TileLang-Ascend 2D RMSNorm kernel: Y = rmsnorm(X) * gamma。

    Args:
        M     : 行数 (必须为偶数 —— 双 Vector 核每核 1 行, wrapper 负责 pad)
        D     : 特征维 (一行整体驻留 UB)
        dtype : 输入/输出精度, 默认 fp16
        eps   : 防除零常数

    out_idx=[-1] : 返回 Y (最后一个 Tensor 参数)
    """
    assert M % 2 == 0, f"M={M} 必须为偶数 (双 Vector 核拆分)"
    num_blocks = M // 2
    inv_d = 1.0 / D  # Python 常量, 编译期折算, 避免核内整型→浮点

    @T.prim_func
    def main(X: T.Tensor((M, D), dtype),
             GAMMA: T.Tensor((1, D), dtype),
             Y: T.Tensor((M, D), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                row = cid * 2 + vid   # 双 Vector 核各处理一行

                # ---- UB 缓冲 (shape 全部带行维 1, 与 reduce/broadcast 对齐) ----
                x_ub = T.alloc_ub((1, D), dtype)
                g_ub = T.alloc_ub((1, D), dtype)
                y_ub = T.alloc_ub((1, D), dtype)
                xf   = T.alloc_ub((1, D), "float32")
                gf   = T.alloc_ub((1, D), "float32")
                yf   = T.alloc_ub((1, D), "float32")
                sq   = T.alloc_ub((1, D), "float32")
                red  = T.alloc_ub((1, 1), "float32")
                inv1 = T.alloc_ub((1, 1), "float32")
                invv = T.alloc_ub((1, D), "float32")

                # ---- GM → UB ----
                T.copy(X[row, 0], x_ub)
                T.copy(GAMMA[0, 0], g_ub)
                T.barrier_all()

                # ---- ① 平方: fp16 cast 到 fp32 再乘 ----
                T.tile.cast(xf, x_ub, "CAST_NONE", D)
                T.tile.cast(gf, g_ub, "CAST_NONE", D)
                T.tile.mul(sq, xf, xf)
                # ---- ② 归约 Σx² → ×1/D → +eps (fp32 累加, "存窄算宽") ----
                T.reduce_sum(sq, red, dim=-1)
                T.tile.mul(red, red, inv_d)
                T.tile.add(red, red, eps)
                # ---- ③ rsqrt 取倒数 (一次标量, 广播回整行) ----
                T.tile.rsqrt(inv1, red)
                T.tile.broadcast(invv, inv1)
                # ---- ④ y = x · inv_rms · gamma (乘倒数, 不做逐元素除) ----
                T.tile.mul(yf, xf, invv)
                T.tile.mul(yf, yf, gf)

                T.tile.cast(y_ub, yf, "CAST_RINT", D)
                T.barrier_all()
                # ---- UB → GM ----
                T.copy(y_ub, Y[row, 0])

    return main


def rmsnorm_tilelang(x, gamma=None, eps: float = 1e-6):
    """
    便捷封装: 1D/2D fp16/fp32 输入 (numpy 或 torch NPU) → 沿最后一维 RMSNorm 输出。

    - gamma 缺省为全 1 (等价于不带缩放的 RMSNorm);
    - 内部把数据放到 NPU 上跑 kernel; 输入 numpy → 返回 numpy, 输入 torch → 返回 torch。
    """
    # ---- 统一转成 numpy, 保存原始 shape/dtype ----
    orig_shape = None
    orig_dtype = None
    is_torch = False
    try:
        import torch
        if isinstance(x, torch.Tensor):
            is_torch = True
            orig_shape = tuple(x.shape)
            orig_dtype = x.dtype
            x = x.detach().cpu().numpy()
            if isinstance(gamma, torch.Tensor):
                gamma = gamma.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        if orig_shape is None:
            orig_shape = x.shape
            orig_dtype = x.dtype
        x = np.asarray(x)
    else:
        raise TypeError(f"不支持的输入类型: {type(x)}, 需要 numpy.ndarray 或 torch.Tensor")

    D = orig_shape[-1]
    batch_shape = orig_shape[:-1]
    M = int(np.prod(batch_shape)) if batch_shape else 1
    x2d = x.reshape(M, D).astype(np.float16)

    if gamma is None:
        gamma = np.ones(D, dtype=np.float16)
    gamma = np.asarray(gamma).astype(np.float16).reshape(1, D)
    assert gamma.shape[1] == D, f"gamma 长度 {gamma.shape[1]} != 特征维 {D}"

    # M pad 到偶数 (双 Vector 核每核一行), 多余行算完丢弃
    M_pad = M + (M % 2)
    if M_pad != M:
        x2d = np.concatenate([x2d, x2d[-1:].repeat(M_pad - M, axis=0)], axis=0)

    import torch
    kernel = rmsnorm_2d(M_pad, D, dtype="float16", eps=eps)
    x_dev = torch.from_numpy(np.ascontiguousarray(x2d)).npu()
    g_dev = torch.from_numpy(np.ascontiguousarray(gamma)).npu()
    y_dev = kernel(x_dev, g_dev)               # (M_pad, D) fp16, NPU
    torch.npu.synchronize()
    y2d = y_dev.cpu().numpy()[:M]

    y = y2d.reshape(orig_shape)
    if is_torch:
        return torch.from_numpy(y).to(orig_dtype)
    if isinstance(orig_dtype, np.dtype):
        return y.astype(orig_dtype)
    return y.astype(np.float16)


# 独立的 numpy ground truth, 与 examples/python/src/rmsnorm.py 保持同公式
def rmsnorm_reference_numpy(x_np: np.ndarray, gamma_np: np.ndarray,
                            eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    g = np.asarray(gamma_np).astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    y = x * inv_rms * g
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    D = 512
    x = (rng.standard_normal(D) * 2.0).astype(np.float16)
    gamma = rng.uniform(0.5, 2.0, D).astype(np.float16)
    y = rmsnorm_tilelang(x, gamma)
    ref = rmsnorm_reference_numpy(x, gamma)
    max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-rmsnorm smoke: D={D}, max_abs_err={max_err:.6e}")
    assert max_err < 1e-2, f"tilelang rmsnorm smoke FAIL, max_err={max_err}"
    print("tilelang-rmsnorm smoke PASSED")
