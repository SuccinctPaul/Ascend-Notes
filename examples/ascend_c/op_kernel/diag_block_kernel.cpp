// 诊断：每个启动的 block 把自己的 (GetBlockIdx(), GetBlockNum()) 通过 GlobalTensor<int32> *diag 写回
// host 分配 N=256 slots for block_bid[N], blk_num[N]  (只要有写就是 block 真启动过)
// 还写: y[bid] = x[bid] 对应 identity (能正确找到自己对应的 bid 才行)
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void diag_block_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // tiling layout (all uint32):
    //   [0] N_elems
    //   [1] diag_slot_count  (= max numBlocks we can record, e.g. 256)
    // workspace: int32 diag[diag_slot_count][2]  (== [block_bid, block_num])
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];
    const uint32_t MAX_DIAG = T[1];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    // workspace as GlobalTensor<int32>: slot 0..2*MAX_DIAG-1   [2*i] = bid [2*i+1] = total
    GlobalTensor<int32_t> Diag;
    Diag.SetGlobalBuffer((__gm__ int32_t*)workspace, (uint64_t)2u * MAX_DIAG);

    const uint32_t bid = GetBlockIdx();
    const uint32_t tot = GetBlockNum();

    // --- 写诊断 ---
    // 把自己 bid 写入 diag[bid*2..], 如果 bid >= MAX_DIAG 就写到 diag[MAX_DIAG-1]
    uint32_t slot = (bid < MAX_DIAG) ? bid : (MAX_DIAG - 1u);
    Diag.SetValue((uint64_t)slot * 2ull,     (int32_t)bid);
    Diag.SetValue((uint64_t)slot * 2ull + 1u, (int32_t)tot);

    // --- 身份: 只有 block bid 处理 index=bid 的元素 (grid-stride 也走)
    const uint64_t stride = (uint64_t)tot;
    for (uint64_t i = (uint64_t)bid; i < (uint64_t)N; i += stride) {
        Yg.SetValue(i, Xg.GetValue(i));
    }
}
