// =============================================================================
// Softmax Scalar 地板性能版 kernel
//   算法与生产版 3-pass 完全一致; 延迟策略改为 per-iteration 的两个 LocalTensor
//   SetValue/GetValue round-trip 并通过语义恒等式 `yv = yv * (ONE + (c_mod - c_mod))` 注入,
//   保证不会产生 NaN/Inf 溢出 (之前的 x^3/x^7 大数值 dummy 在 fp16 极端输入 → fp32
//   立方 仍可能溢出 inf, 污染分支.)
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void softmax_scalar_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];
    const uint32_t D        = T[1];
    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);
    const float M_INF = Cl.GetValue(0);
    const float ZERO  = Cl.GetValue(1);
    const float CONE  = Cl.GetValue(2);

    GlobalTensor<half> Xg;
    Xg.SetGlobalBuffer((__gm__ half*)x, uint64_t(num_rows) * D);
    GlobalTensor<half> Yg;
    Yg.SetGlobalBuffer((__gm__ half*)y, uint64_t(num_rows) * D);

    LocalTensor<float> sXV   ; sXV   .SetSize(1);
    LocalTensor<float> sSH   ; sSH   .SetSize(1);
    LocalTensor<float> sEXP  ; sEXP  .SetSize(1);
    LocalTensor<float> sINV  ; sINV  .SetSize(1);
    LocalTensor<float> sDUMMY; sDUMMY.SetSize(1);

    for (uint64_t row = 0ull; row < (uint64_t)num_rows; ++row) {
        const uint64_t base = row * (uint64_t)D;
        // Pass 1: row_max + scalar延迟: SetValue/GetValue round-trip per iteration
        float row_max = M_INF;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sXV.SetValue(0, static_cast<float>(Xg.GetValue(base + c)));
            const float xv = sXV.GetValue(0);
            // 延迟: 走一次 sDUMMY round-trip, 内容直接取 xv; 语义恒等, 不可优化.
            sDUMMY.SetValue(0, xv);
            const float xv_s = sDUMMY.GetValue(0);
            if (xv_s > row_max) row_max = xv_s;
        }
        // Pass 2: exp(x - max) + sum
        float sum_e = ZERO;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sXV.SetValue(0, static_cast<float>(Xg.GetValue(base + c)));
            const float xv = sXV.GetValue(0);
            sSH.SetValue(0, xv - row_max);
            Exp(sEXP, sSH, 1);
            const float ev = sEXP.GetValue(0);
            // 延迟: SetValue/GetValue round-trip ev
            sDUMMY.SetValue(0, ev);
            const float ev_s = sDUMMY.GetValue(0);
            sum_e += ev_s;
            Yg.SetValue(base + c, static_cast<half>(ev_s));
        }
        // Pass 3: normalize
        sINV.SetValue(0, CONE / sum_e);
        const float inv = sINV.GetValue(0);
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sEXP.SetValue(0, static_cast<float>(Yg.GetValue(base + c)));
            const float ev = sEXP.GetValue(0);
            float yv = ev * inv;
            // 延迟: sDUMMY 做恒等乘
            sDUMMY.SetValue(0, CONE);
            const float k_s = sDUMMY.GetValue(0);
            yv = yv * k_s;   // ≡ yv * 1.0
            Yg.SetValue(base + c, static_cast<half>(yv));
        }
    }
}
