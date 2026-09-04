"""
RoPE kernel —— TileLang + tilelang-ascend 后端 (交错配对旋转)。

    对 token 向量 x 拆成 D/2 对相邻分量 (交错配对, RoFormer 原版):
        y[2a]   = x[2a]·cos_a - x[2a+1]·sin_a
        y[2a+1] = x[2a]·sin_a + x[2a+1]·cos_a

设计要点 (docs/04 §5.2 §5.3):
  - cos/sin 表由 host 预计算 (查表 + 乘加, 别在 kernel 里现算三角);
  - 纯逐元素, 无归约, T.serial 逐对二维旋转 (复数乘), fp16 精度;
  - 写法约定与官方 elementwise 示例一致: (cid, vid) 双 Vector 核 + T.Scope("V")
    + T.barrier_all, kernel 接收 torch NPU 张量, 2D 一次 launch 处理所有行。

封装器 `rope_tilelang` 接受 (M, D) 输入 + 每行位置 positions (M,)。
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
def rope_2d(M: int, D: int, dtype: str = "float16"):
    """
    TileLang-Ascend 2D RoPE kernel: Y = rotate(X, COS, SIN)。

    Args:
        M     : token 行数 (必须为偶数 —— 双 Vector 核每核 1 行, wrapper 负责 pad)
        D     : 向量维 (偶数, D == 2*HALF)
        dtype : 输入/输出精度, 默认 fp16

    out_idx=[-1] : 返回 Y (最后一个 Tensor 参数)
    """
    assert M % 2 == 0, f"M={M} 必须为偶数 (双 Vector 核拆分)"
    assert D % 2 == 0, f"RoPE 要求 D 为偶数, 得到 D={D}"
    HALF = D // 2
    num_blocks = M // 2

    @T.prim_func
    def main(X: T.Tensor((M, D), dtype),
             COS: T.Tensor((M, HALF), dtype),
             SIN: T.Tensor((M, HALF), dtype),
             Y: T.Tensor((M, D), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                row = cid * 2 + vid   # 双 Vector 核各处理一个 token

                # ---- UB 缓冲: x / cos/sin 表 / 输出 y ----
                x_ub   = T.alloc_ub((1, D), dtype)
                cos_ub = T.alloc_ub((1, HALF), dtype)
                sin_ub = T.alloc_ub((1, HALF), dtype)
                y_ub   = T.alloc_ub((1, D), dtype)

                # ---- GM → UB: x + 该行 cos/sin 表 ----
                T.copy(X[row, 0], x_ub)
                T.copy(COS[row, 0], cos_ub)
                T.copy(SIN[row, 0], sin_ub)
                T.barrier_all()

                # ---- 逐对二维旋转 (复数乘): 交错配对 (x[2a], x[2a+1]) ----
                # aicore 禁止 fp16 标量算术 → 显式升 fp32 做乘加, 再降回 fp16 存
                # (这正是 docs/04 §5 的 "fp32 中间量" 原则)
                for a in T.serial(HALF):
                    x1 = x_ub[0, 2 * a].astype("float32")
                    x2 = x_ub[0, 2 * a + 1].astype("float32")
                    c  = cos_ub[0, a].astype("float32")
                    s  = sin_ub[0, a].astype("float32")
                    y_ub[0, 2 * a]     = (x1 * c - x2 * s).astype("float16")
                    y_ub[0, 2 * a + 1] = (x1 * s + x2 * c).astype("float16")

                T.barrier_all()
                # ---- UB → GM ----
                T.copy(y_ub, Y[row, 0])

    return main


def precompute_rope_tables(positions: np.ndarray, d: int,
                           base: float = 10000.0):
    """host 侧预计算 cos/sin 表 (T, d/2), fp32 → kernel 前转 fp16."""
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    angles = np.asarray(positions, dtype=np.float32)[..., None] * inv_freq
    return np.cos(angles), np.sin(angles)


def rope_tilelang(x, positions=None, base: float = 10000.0):
    """
    便捷封装: 1D/2D fp16/fp32 输入 (numpy 或 torch NPU) → 沿最后一维 RoPE 旋转。

    - 对 1D (D,): positions 缺省 [0]
    - 对 2D (M, D): positions 缺省 [0, 1, ..., M-1]; 也可显式传入长度 M 的数组
    - D 为奇数时报错 (RoPE 语义要求偶数维)
    - 输入 numpy → 返回 numpy, 输入 torch → 返回 torch。
    """
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
    if D % 2 != 0:
        raise ValueError(f"RoPE 要求最后一维为偶数, 得到 D={D}")
    HALF = D // 2
    batch_shape = orig_shape[:-1]
    M = int(np.prod(batch_shape)) if batch_shape else 1
    x2d = x.reshape(M, D).astype(np.float16)

    if positions is None:
        positions = np.arange(M)
    positions = np.asarray(positions).reshape(-1)
    assert positions.shape[0] == M, f"positions 长度 {positions.shape[0]} != 行数 {M}"

    cos_np, sin_np = precompute_rope_tables(positions, D, base)
    cos16 = np.ascontiguousarray(cos_np.astype(np.float16))
    sin16 = np.ascontiguousarray(sin_np.astype(np.float16))

    # M pad 到偶数 (双 Vector 核每核一行), 多余行算完丢弃
    M_pad = M + (M % 2)
    if M_pad != M:
        x2d = np.concatenate([x2d, x2d[-1:].repeat(M_pad - M, axis=0)], axis=0)
        cos16 = np.concatenate([cos16, cos16[-1:].repeat(M_pad - M, axis=0)], axis=0)
        sin16 = np.concatenate([sin16, sin16[-1:].repeat(M_pad - M, axis=0)], axis=0)

    import torch
    kernel = rope_2d(M_pad, D, dtype="float16")
    x_dev = torch.from_numpy(np.ascontiguousarray(x2d)).npu()
    c_dev = torch.from_numpy(cos16).npu()
    s_dev = torch.from_numpy(sin16).npu()
    y_dev = kernel(x_dev, c_dev, s_dev)        # (M_pad, D) fp16, NPU
    torch.npu.synchronize()
    y2d = y_dev.cpu().numpy()[:M]

    y = y2d.reshape(orig_shape)
    if is_torch:
        return torch.from_numpy(y).to(orig_dtype)
    if isinstance(orig_dtype, np.dtype):
        return y.astype(orig_dtype)
    return y.astype(np.float16)


# 独立的 numpy ground truth, 与 examples/python/src/rope.py 保持同公式
def rope_reference_numpy(x_np: np.ndarray, positions: np.ndarray,
                         base: float = 10000.0) -> np.ndarray:
    x = np.asarray(x_np).astype(np.float32)
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    angles = np.asarray(positions, dtype=np.float32)[..., None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y = np.empty_like(x)
    y[..., 0::2] = x1 * cos - x2 * sin
    y[..., 1::2] = x1 * sin + x2 * cos
    return y.astype(x_np.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    D = 512
    x = (rng.standard_normal(D) * 2.0).astype(np.float16)
    pos = np.array([7])
    y = rope_tilelang(x, pos)
    ref = rope_reference_numpy(x, pos)
    max_err = float(np.max(np.abs(y.astype(np.float32) - ref.astype(np.float32))))
    print(f"tilelang-rope smoke: D={D}, max_abs_err={max_err:.6e}")
    assert max_err < 1e-2, f"tilelang rope smoke FAIL, max_err={max_err}"
    print("tilelang-rope smoke PASSED")
