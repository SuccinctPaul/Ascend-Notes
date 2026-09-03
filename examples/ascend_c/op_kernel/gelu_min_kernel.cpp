// Super-minimal GELU kernel: 完全跳过 DataCopy / LocalTensor tile / Vector tile
// 只用 GlobalTensor::GetValue / SetValue + Vector Exp(Size=1, count=1) 实现
// y = x / (1 + exp(-2*inner)).  若该 kernel PASS → 必然是 DataCopy/LocalTensor<256> 出的错.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_min_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    // 1-slot 缓冲 for Vector Exp
    LocalTensor<float> sArg; sArg.SetSize(1);
    LocalTensor<float> sExp; sExp.SetSize(1);

    const float CSQRT = 0.7978845608028654f;
    const float CCUB  = 0.044715f;
    const float TWO   = 2.0f;
    const float ONE   = 1.0f;

    const int64_t N64 = (int64_t)N;
    const int32_t stride = (int32_t)GetBlockNum();
    const int32_t start  = (int32_t)GetBlockIdx();

    for (int64_t i = start; i < N64; i += (int64_t)stride) {
        const float xv = static_cast<float>(Xg.GetValue(i));
        const float x3 = xv * xv * xv;
        const float inner = CSQRT * (xv + CCUB * x3);
        const float neg_two_inner = (ONE - ONE) - TWO * inner;
        sArg.SetValue(0, neg_two_inner);
        Exp(sExp, sArg, 1);
        const float den = ONE + sExp.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
