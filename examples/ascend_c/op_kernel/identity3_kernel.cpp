// IDENTITY3: 验证 GlobalTensor 的 indexing 到底是元素 offset 还是字节 offset.
// 每个 block bid 只写 index==bid 的位置, 用多种表达式:
//   mode=0: 直接 i = bid (元素 offset, 预期 PASS)
//   mode=1: i = bid * sizeof(half) (字节 offset, 若这才对才 PASS)
//   mode=2: 单 block 全串行 (再确认和之前 mode=3 一样 PASS)
//   mode=3: bid = (int32_t)GetBlockIdx(),  i = (int64_t)bid
// 另外我们额外把 "block bid 收到什么 bid 值" 写入 y[bid + N] (N 作为 diag 区, 需要 N+16 <= 总长度)
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void identity3_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N    = t[0];
    const uint32_t mode = t[1];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N + 16u);   // +16 for diag storage
    Yg.SetGlobalBuffer((__gm__ half*)y, N + 16u);

    const uint32_t bid_u32 = GetBlockIdx();
    const uint32_t blk_n   = GetBlockNum();

    // diag: y[N + bid] = half(-1.0) 告知 host block bid 实际有执行
    if (bid_u32 < 16u) {
        const float marker = -1.0f;   // marker = "block ran OK"
        Yg.SetValue((uint64_t)N + bid_u32, half(marker));
    }

    if (mode == 0u) {
        // block bid 只处理 index == bid_u32 这一个元素: element offset
        if (bid_u32 < N) {
            half v = Xg.GetValue((uint64_t)bid_u32);
            Yg.SetValue((uint64_t)bid_u32, v);
        }
    } else if (mode == 1u) {
        // 假设 GlobalTensor index 实际是字节 offset, 则元素 k 需要 index = k * 2
        if (bid_u32 < N) {
            uint64_t byte_idx = (uint64_t)bid_u32 * 2ull;
            half v = Xg.GetValue(byte_idx);
            Yg.SetValue(byte_idx, v);
        }
    } else if (mode == 2u) {
        // block 0 全串行 (和 mode=3 之前的 PASS 对照)
        if (bid_u32 == 0u) {
            for (uint64_t i = 0ull; i < (uint64_t)N; ++i) {
                Yg.SetValue(i, Xg.GetValue(i));
            }
        }
    } else if (mode == 3u) {
        // 同 mode=0 但 index = (int64_t)(int32_t)bid  (模仿原先 int32_t → int64_t cast 的代码)
        const int32_t bid_i32 = (int32_t)bid_u32;
        const int64_t idx64   = (int64_t)bid_i32;
        if (bid_u32 < N) {
            Yg.SetValue(idx64, Xg.GetValue(idx64));
        }
    }
    (void)blk_n;
}
