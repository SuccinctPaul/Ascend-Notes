// =============================================================================
// GELU Scalar 地板性能版 kernel — Ascend C
//
// 算法与 gelu_kernel (v6) 完全一致: GELU(x) = x / (1 + exp(-CBIG*(x+CCUB*x^3))).
// 延迟策略: per-iteration 增加一次 LocalTensor<float> round-trip (SetValue + GetValue)
// 并通过恒等乘 CONE 注入; 绝不构造 x^4 / x^7 这种大数值临时变量 (fp16 极端输入
// 下仍会溢出 inf → 语义恒等变形变成 yv + inf = inf 破坏正确性).
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_scalar_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N_u32 = T[0];

    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 2), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);
    const float CBIG = Cl.GetValue(0);
    const float CCUB = Cl.GetValue(1);
    const float CONE = Cl.GetValue(2);

    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N_u32);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N_u32);

    LocalTensor<float> sXV   ; sXV   .SetSize(1);
    LocalTensor<float> sSH   ; sSH   .SetSize(1);
    LocalTensor<float> sEXP  ; sEXP  .SetSize(1);
    LocalTensor<float> sDUMMY; sDUMMY.SetSize(1);

    const uint64_t N = (uint64_t)N_u32;
    for (uint64_t i = 0ull; i < N; ++i) {
        sXV.SetValue(0, static_cast<float>(Xg.GetValue(i)));
        const float xv = sXV.GetValue(0);
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float pos = CBIG * t1;
        const float big = xv + pos;
        sSH.SetValue(0, sXV.GetValue(0) - big);
        Exp(sEXP, sSH, 1);
        const float den = CONE + sEXP.GetValue(0);
        float yv = xv / den;
        // ---- Scalar 延迟: 两次 SetValue/GetValue round-trip + 恒等乘/加 ----
        sDUMMY.SetValue(0, CONE);
        const float k1 = sDUMMY.GetValue(0);
        sDUMMY.SetValue(0, 0.0f);
        const float k0 = sDUMMY.GetValue(0);
        yv = yv * k1 + k0;  // ≡ yv*1 + 0 = yv
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
