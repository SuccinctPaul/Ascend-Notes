// V3 GELU kernel: scalar 风格, 但使用已验证通过的 "数据流 fsub" pattern.
// 这是一个独立 kernel, 函数名叫 gelu_v3_kernel, 对应 target ascend_gelu_v3 (新 target)
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_v3_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<float> sCBIG; sCBIG.SetSize(1); sCBIG.SetValue(0, 1.5957691216057308f);  // 2*sqrt(2/pi)
    LocalTensor<float> sCCUB; sCCUB.SetSize(1); sCCUB.SetValue(0, 0.044715f);
    LocalTensor<float> sCONE; sCONE.SetSize(1); sCONE.SetValue(0, 1.0f);
    LocalTensor<float> sARG ; sARG .SetSize(1);
    LocalTensor<float> sEXP ; sEXP .SetSize(1);
    LocalTensor<float> sBIG ; sBIG .SetSize(1);
    const float CBIG = sCBIG.GetValue(0);
    const float CCUB = sCCUB.GetValue(0);
    const float CONE = sCONE.GetValue(0);

    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        const float xv = static_cast<float>(Xg.GetValue(i));
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float pos = CBIG * t1;
        // arg = -pos, 构造 "x_value - roundtrip(x_value + pos)" 形式.
        const float shifted = xv + pos;
        sBIG.SetValue(0, shifted);
        const float big = sBIG.GetValue(0);
        const float raw_diff = xv - big;   // === -pos (数学上)
        // 放入 LocalTensor 再读出 — 完全 mirror softmax 的 sArg 生命周期
        sARG.SetValue(0, raw_diff);
        const float arg = sARG.GetValue(0);
        // arg in sARG
        sARG.SetValue(0, arg);
        Exp(sEXP, sARG, 1);
        const float den = CONE + sEXP.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
