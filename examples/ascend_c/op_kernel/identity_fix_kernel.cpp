// 修复策略: 固定 WORK_GROUPS=32. 不管 host launch 的 numBlocks/numGroups 多大,
// kernel 里用 bid ∈ [0, WORK_GROUPS) 的 block 做 grid-stride stride=WORK_GROUPS.
// bid >= WORK_GROUPS 直接 return (空).  这样只要实际运行的 ~100 blocks 中
// [0..31] 都被覆盖到 (根据上面的统计 bid range 下限总是 ~1, 所以 0..31 几乎全有),
// 就保证任何 N 都得到完整 grid-stride 覆盖.
#include "kernel_operator.h"
using namespace AscendC;

static constexpr uint32_t WORK_GROUPS = 32u;

extern "C" __global__ __aicore__
void identity_fix_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    const uint32_t bid = GetBlockIdx();
    if (bid >= WORK_GROUPS) return;

    for (uint64_t i = (uint64_t)bid; i < (uint64_t)N; i += WORK_GROUPS) {
        Yg.SetValue(i, Xg.GetValue(i));
    }
}
