// 复用 identity3_kernel mode=2 的逻辑, 直接写一个始终 "block 0 处理全部" 的 kernel.
// 这样即便 host 端还传错 numBlocks, 也只有 block 0 在干真正的事, 且干的是全串行的活.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void identity_s0_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);
    if (GetBlockIdx() == 0u) {
        for (uint64_t i = 0ull; i < (uint64_t)N; ++i) {
            Yg.SetValue(i, Xg.GetValue(i));
        }
    }
}
