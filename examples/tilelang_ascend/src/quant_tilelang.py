"""
INT8 对称量化 kernel —— TileLang + tilelang-ascend 后端。

对应 docs/ops/08-quantization.md §2.1 的对称量化 (per-row scale, §5.3):

    quant:   scale[r] = absmax(x[r,:]) / 127              (§2.1 公式)
             q[r,:]   = clamp_round(x[r,:] / scale[r]) → int8
    dequant: x̂[r,:]   = q[r,:] * scale[r]                  (Vector 上做, §5.4)

实现 (写法约定同 rmsnorm_tilelang.py): 2D kernel, (cid, vid) 双 Vector 核每核一行:
    T.tile.cast   : fp16 → fp32 (归约/缩放在宽精度)
    T.tile.abs + T.reduce_max : 逐行 absmax
    T.tile.div    : x / scale (scale=broadcast(absmax/127))
    T.tile.cast   : fp32 → int8 (CAST_RINT = round-to-nearest-even)
    dequant 方向  : int8 → fp32 cast, 乘 broadcast scale, cast 回 fp16

kernel 接收 torch NPU 张量 (cython adapter 约定, 见 rmsnorm_tilelang.py);
SCALE 由 caller 分配传入, kernel 原地写 (side-effect)。
"""

import numpy as np

import tilelang
import tilelang.language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def quant_2d(M: int, D: int):
    """
    逐行 absmax 对称量化 kernel: X (M,D) fp16 → Q (M,D) int8 + SCALE (M,1) fp32。

    SCALE 由 caller 分配并传入, kernel 原地写; Q 通过 out_idx 由 adapter 返回。
    """
    assert M % 2 == 0, f"M={M} 必须为偶数 (双 Vector 核拆分)"
    num_blocks = M // 2

    @T.prim_func
    def main(X: T.Tensor((M, D), "float16"),
             SCALE: T.Tensor((M, 1), "float32"),
             Q: T.Tensor((M, D), "int8")):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                row = cid * 2 + vid

                x_ub = T.alloc_ub((1, D), "float16")
                xf   = T.alloc_ub((1, D), "float32")
                ax   = T.alloc_ub((1, D), "float32")
                red  = T.alloc_ub((1, 1), "float32")
                sc   = T.alloc_ub((1, 1), "float32")
                c127 = T.alloc_ub((1, 1), "float32")
                ceps = T.alloc_ub((1, 1), "float32")
                scb  = T.alloc_ub((1, D), "float32")
                xh16 = T.alloc_ub((1, D), "float16")
                q8   = T.alloc_ub((1, D), "int8")

                # ---- GM → UB ----
                T.copy(X[row, 0], x_ub)
                T.barrier_all()

                # ---- scale = max(absmax/127, eps) ----
                T.tile.cast(xf, x_ub, "CAST_NONE", D)
                T.tile.abs(ax, xf)
                T.reduce_max(ax, red, dim=-1)
                T.tile.fill(c127, 127.0)
                T.tile.div(sc, red, c127)          # absmax / 127
                T.tile.fill(ceps, 1e-12)
                T.tile.max(sc, sc, ceps)           # 全零行防除零
                T.copy(sc, SCALE[row, 0])          # scale 下发 GM (fp32)

                # ---- q = cast_round(x / scale) → int8 ----
                T.tile.broadcast(scb, sc)
                T.tile.div(xf, xf, scb)            # x / scale ∈ [-127, 127]
                # dav-c220 无 fp32↔int8 直转: fp32 → fp16 → int8 (CAST_RINT)
                T.tile.cast(xh16, xf, "CAST_NONE", D)
                T.tile.cast(q8, xh16, "CAST_RINT", D)
                T.barrier_all()
                T.copy(q8, Q[row, 0])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def dequant_2d(M: int, D: int, dtype: str = "float16"):
    """反量化 kernel: Q (M,D) int8 × SCALE (M,1) fp32 → Y (M,D) fp16 (§5.4 Vector 反量化)."""
    assert M % 2 == 0, f"M={M} 必须为偶数 (双 Vector 核拆分)"
    num_blocks = M // 2

    @T.prim_func
    def main(Q: T.Tensor((M, D), "int8"),
             SCALE: T.Tensor((M, 1), "float32"),
             Y: T.Tensor((M, D), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                row = cid * 2 + vid

                q8  = T.alloc_ub((1, D), "int8")
                qh  = T.alloc_ub((1, D), "float16")
                qf  = T.alloc_ub((1, D), "float32")
                sc  = T.alloc_ub((1, 1), "float32")
                scb = T.alloc_ub((1, D), "float32")
                yf  = T.alloc_ub((1, D), "float32")
                y_ub = T.alloc_ub((1, D), dtype)

                T.copy(Q[row, 0], q8)
                T.copy(SCALE[row, 0], sc)
                T.barrier_all()

                # dav-c220 无 fp32↔int8 直转: int8 → fp16 → fp32
                T.tile.cast(qh, q8, "CAST_NONE", D)
                T.tile.cast(qf, qh, "CAST_NONE", D)
                T.tile.broadcast(scb, sc)
                T.tile.mul(yf, qf, scb)                  # × scale
                T.tile.cast(y_ub, yf, "CAST_RINT", D)    # fp32 → fp16

                T.barrier_all()
                T.copy(y_ub, Y[row, 0])

    return main


def quant_int8_tilelang(x):
    """
    便捷封装: (M, D) fp16 numpy/torch → (q int8, scale fp32)。

    M pad 到偶数 (双 Vector 核); 输入 numpy → 输出 numpy, 输入 torch → 输出 torch。
    """
    orig_shape = None
    is_torch = False
    try:
        import torch
        if isinstance(x, torch.Tensor):
            is_torch = True
            orig_shape = tuple(x.shape)
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        if orig_shape is None:
            orig_shape = x.shape
        x = np.asarray(x)
    else:
        raise TypeError(f"不支持的输入类型: {type(x)}")

    D = orig_shape[-1]
    M = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
    x2d = x.reshape(M, D).astype(np.float16)

    M_pad = M + (M % 2)
    if M_pad != M:
        x2d = np.concatenate([x2d, x2d[-1:].repeat(M_pad - M, axis=0)], axis=0)

    import torch
    kernel = quant_2d(M_pad, D)
    x_dev = torch.from_numpy(np.ascontiguousarray(x2d)).npu()
    scale_dev = torch.zeros((M_pad, 1), dtype=torch.float32, device="npu")
    q_dev = kernel(x_dev, scale_dev)            # out_idx=[-1] → Q
    torch.npu.synchronize()
    q = q_dev.cpu().numpy()[:M]
    scale = scale_dev.cpu().numpy()[:M, 0]

    if is_torch:
        return (torch.from_numpy(q), torch.from_numpy(scale))
    return q, scale


def dequant_int8_tilelang(q, scale):
    """反量化封装: (q int8, scale fp32) → fp16, shape/dtype 约定同 quant_int8_tilelang."""
    orig_shape = None
    is_torch = False
    try:
        import torch
        if isinstance(q, torch.Tensor):
            is_torch = True
            orig_shape = tuple(q.shape)
            q = q.detach().cpu().numpy()
            if isinstance(scale, torch.Tensor):
                scale = scale.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(q, np.ndarray):
        if orig_shape is None:
            orig_shape = q.shape
        q = np.asarray(q)
    else:
        raise TypeError(f"不支持的输入类型: {type(q)}")

    D = orig_shape[-1]
    M = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
    q2d = q.reshape(M, D)
    scale2d = np.asarray(scale, dtype=np.float32).reshape(M, 1)

    M_pad = M + (M % 2)
    if M_pad != M:
        q2d = np.concatenate([q2d, np.zeros((M_pad - M, D), dtype=np.int8)], axis=0)
        scale2d = np.concatenate([scale2d, np.zeros((M_pad - M, 1), dtype=np.float32)], axis=0)

    import torch
    kernel = dequant_2d(M_pad, D, dtype="float16")
    q_dev = torch.from_numpy(np.ascontiguousarray(q2d)).npu()
    s_dev = torch.from_numpy(np.ascontiguousarray(scale2d)).npu()
    y_dev = kernel(q_dev, s_dev)
    torch.npu.synchronize()
    y = y_dev.cpu().numpy()[:M].reshape(orig_shape)

    if is_torch:
        return torch.from_numpy(y)
    return y


# Ground truth (numpy), 与 examples/python/src/quant.py 同公式
def quant_int8_reference_numpy(x_np: np.ndarray):
    x = np.asarray(x_np).astype(np.float32)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    amax = np.maximum(amax, 1e-12)
    scale = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(x.shape[:-1])


def dequant_int8_reference_numpy(q_np: np.ndarray, scale_np: np.ndarray) -> np.ndarray:
    qf = np.asarray(q_np).astype(np.float32)
    return (qf * np.asarray(scale_np, dtype=np.float32)[..., None]).astype(np.float16)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    M, D = 8, 512
    x = (rng.standard_normal((M, D)) * 2.0).astype(np.float16)
    q, scale = quant_int8_tilelang(x)
    q_ref, scale_ref = quant_int8_reference_numpy(x)
    q_match = float(np.mean(q.astype(np.int32) == q_ref.astype(np.int32)))
    scale_err = float(np.max(np.abs(scale - scale_ref)))
    y = dequant_int8_tilelang(q, scale)
    y_ref = dequant_int8_reference_numpy(q_ref, scale_ref)
    rt = float(np.max(np.abs(y.astype(np.float32) - x.astype(np.float32))))
    print(f"tilelang-quant smoke M={M} D={D}: q_match={q_match:.4f} "
          f"scale_err={scale_err:.2e} roundtrip={rt:.6f} (上界 {float(scale_ref.max()):.6f})")
    # q 允许 ±1 LSB (fp16 中转舍入, dav-c220 无 fp32↔int8 直转); 硬标准是往返误差
    assert q_match > 0.98 and rt <= float(scale_ref.max()) + 1e-6
    print("tilelang-quant smoke PASSED")
