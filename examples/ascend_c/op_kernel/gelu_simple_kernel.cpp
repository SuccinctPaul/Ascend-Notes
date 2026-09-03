// =============================================================================
// 简化版 GELU 生产 kernel:
//   · 保持 TILE=256 DataCopy → LocalTensor<half>
//   · 计算阶段: 逐元素 scalar GetValue → fp32 运算 (std::tanh) → scalar SetValue
//     (避开 Vector 原语 + LocalTensor cast 的组合问题)
//   · 写回: DataCopy half UB → GM
//
// 目的: 隔离 "Vector API 正确/错误" 与 "DataCopy/索引 正确/错误" 两个维度.
// =============================================================================
#include "kernel_operator.h"
#include <cmath>
using namespace AscendC;

static constexpr int32_t TILE = 256;
static constexpr float  CSQRT = 0.7978845608028654f;
static constexpr float  CCUB  = 0.044715f;

extern "C" __global__ __aicore__
void gelu_simple_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<half> xh_buf; xh_buf.SetSize(TILE);
    LocalTensor<half> yh_buf; yh_buf.SetSize(TILE);

    const uint32_t num_tiles = (N + TILE - 1u) / TILE;
    const uint32_t stride    = GetBlockNum();
    const uint32_t start     = GetBlockIdx();

    for (uint32_t bid = start; bid < num_tiles; bid += stride) {
        const uint32_t base = bid * TILE;
        const uint32_t rem  = (base + TILE <= N) ? TILE : (N - base);

        // 1. DataCopy half GM → UB (完整 tile)
        if (rem == TILE) {
            DataCopy(xh_buf, Xg[base], TILE);
        } else {
            for (int32_t j = 0; j < TILE; ++j) xh_buf.SetValue(j, half(0.0f));
            for (uint32_t j = 0; j < rem; ++j) {
                half v = Xg.GetValue(uint64_t(base) + j);
                xh_buf.SetValue(int32_t(j), v);
            }
        }

        // 2. 逐元素 scalar 计算 GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        for (int32_t j = 0; j < TILE; ++j) {
            const float xv = static_cast<float>(xh_buf.GetValue(j));
            const float x3 = xv * xv * xv;
            const float inner = CSQRT * (xv + CCUB * x3);
            // tanh via std::tanh (bisheng fp32 scalar)
            const float th = tanhf(inner);
            const float yv = xv * 0.5f * (1.0f + th);
            yh_buf.SetValue(j, static_cast<half>(yv));
        }

        // 3. DataCopy half UB → GM
        if (rem == TILE) {
            DataCopy(Yg[base], yh_buf, TILE);
        } else {
            for (uint32_t j = 0; j < rem; ++j) {
                half v = yh_buf.GetValue(int32_t(j));
                Yg.SetValue(uint64_t(base) + j, v);
            }
        }
    }
}
