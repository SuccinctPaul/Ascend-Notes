// Minimal IDENTITY test: y[i] = x[i] 仅用 GlobalTensor scalar
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void identity_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    const int64_t N64 = (int64_t)N;
    const int32_t stride = (int32_t)GetBlockNum();
    const int32_t start  = (int32_t)GetBlockIdx();
    for (int64_t i = start; i < N64; i += (int64_t)stride) {
        half v = Xg.GetValue(i);
        Yg.SetValue(i, v);
    }
}
