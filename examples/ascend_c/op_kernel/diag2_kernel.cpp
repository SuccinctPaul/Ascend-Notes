// =============================================================================
// Diag2: 最简 Vector 管线诊断 —— LocalTensor + DataCopy + Muls(y = 0.5*x)
// 用来定位 Vector core exception 来自: SetSize/DataCopy/Vector 原语 哪一层?
//
// CANN 9.0 官方 API (参考 ge_glu_v2_base.h ComputeGeluBase):
//   Muls(dst, src, scalar, count)       dst[i] = scalar * src[i]
//   Mul(dst, a, b, count)              dst[i] = a[i] * b[i]
//   Adds(dst, src, scalar, count)      dst[i] = src[i] + scalar
//   Add(dst, a, b, count)              dst[i] = a[i] + b[i]
//   Exp(dst, src, count)               dst[i] = exp(src[i])
//   Div(dst, a, b, count)              dst[i] = a[i] / b[i]
//   Cast<To,Ti>(dst, src, count)       dst[i] = (To)src[i]
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

static constexpr int32_t TILE = 256;

extern "C" __global__ __aicore__
void diag2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<half> x_buf;
    LocalTensor<half> y_buf;
    x_buf.SetSize(TILE);
    y_buf.SetSize(TILE);

    // 注意: CANN 9.0 AscendC Muls<T> 的 scalar 参数类型是 float (参考 ComputeGeluBase),
    // 哪怕 T == half, 也要传 float 标量, 否则 half(scalar) 会被当作 1.0 (即 Mul 变成恒等变换)
    const float scale_f = 0.5f;

    const uint32_t num_tiles = (N + TILE - 1u) / TILE;
    const uint32_t stride    = GetBlockNum();
    const uint32_t start     = GetBlockIdx();

    for (uint32_t bid = start; bid < num_tiles; bid += stride) {
        const uint32_t base = bid * TILE;
        const uint32_t rem  = (base + TILE <= N) ? TILE : (N - base);

        if (rem == TILE) {
            DataCopy(x_buf, Xg[base], TILE);
        } else {
            for (int32_t j = 0; j < TILE; ++j) x_buf.SetValue(j, half(0.0f));
            for (uint32_t j = 0; j < rem; ++j)
                x_buf.SetValue((int32_t)j, Xg.GetValue((uint64_t)base + j));
        }

        // y_buf = 0.5 * x_buf  ← scalar=float, 对齐 CANN 官方 ComputeGeluBase 写法
        Muls(y_buf, x_buf, scale_f, TILE);

        if (rem == TILE) {
            DataCopy(Yg[base], y_buf, TILE);
        } else {
            for (uint32_t j = 0; j < rem; ++j)
                Yg.SetValue((uint64_t)base + j, y_buf.GetValue((int32_t)j));
        }
    }
}
