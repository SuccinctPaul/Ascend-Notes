// 诊断专用 kernel: 计算每个元素的 (positive, -positive, exp(arg)) 并写入 y[0..3]
// 固定 N=8.  直接显式做 N=8.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void diag3_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; (void)tiling;
    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, 8u);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, 8u);
    LocalTensor<float> sArg; sArg.SetSize(1);
    LocalTensor<float> sExp; sExp.SetSize(1);
    LocalTensor<float> sCBIG ; sCBIG .SetSize(1); sCBIG .SetValue(0, 1.5957691216057308f);
    LocalTensor<float> sCCUB ; sCCUB .SetSize(1); sCCUB .SetValue(0, 0.044715f);
    const float CBIG = sCBIG.GetValue(0);
    const float CCUB = sCCUB.GetValue(0);

    // 把每个元素的 3 个诊断值 (positive, datapath_neg_positive, exp(arg), den) 写 y[0..23]
    for (uint64_t k = 0ull; k < 8ull; ++k) {
        const float xv = static_cast<float>(Xg.GetValue(k));
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float positive = CBIG * t1;
        const float zero_dp = xv - xv;
        const float arg = zero_dp - positive;
        // y[k*4 + 0] = positive (写回 fp16, 必然溢出/截断但能看数量级)
        Yg.SetValue(k*4 + 0, static_cast<half>(positive));
        Yg.SetValue(k*4 + 1, static_cast<half>(arg));
        sArg.SetValue(0, arg);
        Exp(sExp, sArg, 1);
        Yg.SetValue(k*4 + 2, static_cast<half>(sExp.GetValue(0)));
        const float den = 1.0f + sExp.GetValue(0);
        Yg.SetValue(k*4 + 3, static_cast<half>(den));
    }
}
