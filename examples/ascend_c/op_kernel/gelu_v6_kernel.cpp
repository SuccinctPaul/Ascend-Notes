// GELU v6 kernel: 彻底 mirror softmax 的写法 + 从 DataCopy 读常数.
//  只做最少的 LocalTensor 存取 roundtrip: sXV 仅存 xv, sSH 仅存减法结果.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_v6_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];
    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 2), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);
    const float CBIG = Cl.GetValue(0);  // 2*sqrt(2/pi)
    const float CCUB = Cl.GetValue(1);  // 0.044715
    const float CONE = Cl.GetValue(2);  // 1.0

    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<float> sXV ; sXV .SetSize(1);
    LocalTensor<float> sSH ; sSH .SetSize(1);
    LocalTensor<float> sEXP; sEXP.SetSize(1);

    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        // 1. xv round-trip (同 softmax)
        sXV.SetValue(0, static_cast<float>(Xg.GetValue(i)));
        const float xv = sXV.GetValue(0);
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        // big = xv + CBIG*(xv + CCUB*x^3)
        //       = xv + CBIG*xv + CBIG*CCUB*x^3   (都用 fp32 local var, 同 softmax row_max)
        const float t1 = CCUB * x3;
        const float t2 = xv + t1;
        const float pos = CBIG * t2;
        const float big = xv + pos;
        // 2. 同 softmax: sSH.SetValue(0, sXV.GetValue(0) - local_float_var)
        sSH.SetValue(0, sXV.GetValue(0) - big);   // arg = -pos.  完全同构 sXV.GetValue(0) - row_max.
        // 3. Exp(sEXP, sSH, 1)
        Exp(sEXP, sSH, 1);
        // 4. y = xv / (CONE + exp(arg)).  除法与 v5 之前标量版语义相同.
        const float den = CONE + sEXP.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
