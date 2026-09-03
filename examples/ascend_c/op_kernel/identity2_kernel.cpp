// IDENTITY variants 对比三种 indexing:
//   mode=0  int64  start = GetBlockIdx() (baseline, 可能错)
//   mode=1  int32  start = GetBlockIdx(), int64_t(start) cast 在 LHS
//   mode=2  uint32 start/stride, N as uint32
//   mode=3 单 block (GetBlockNum()==1) 串行处理所有 idx
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void identity2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N    = t[0];
    const uint32_t mode = t[1];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    if (mode == 0u) {
        // --- variant 0: int64 everything (baseline FAIL from before) ---
        const int64_t N64    = (int64_t)N;
        const int32_t stride = (int32_t)GetBlockNum();
        const int32_t start  = (int32_t)GetBlockIdx();
        for (int64_t i = start; i < N64; i += (int64_t)stride) {
            Yg.SetValue(i, Xg.GetValue(i));
        }
    } else if (mode == 1u) {
        // --- variant 1: uint32 start/stride, explicit cast before use ---
        const uint32_t stride = GetBlockNum();
        const uint32_t start  = GetBlockIdx();
        for (uint32_t i = start; i < N; i += stride) {
            Yg.SetValue(i, Xg.GetValue(i));
        }
    } else if (mode == 2u) {
        // --- variant 2: 强制 uint64_t indexing (和 GlobalTensor 的 GetValue 官方签名对齐) ---
        const uint64_t N64    = (uint64_t)N;
        const uint64_t stride = (uint64_t)GetBlockNum();
        const uint64_t start  = (uint64_t)GetBlockIdx();
        for (uint64_t i = start; i < N64; i += stride) {
            Yg.SetValue(i, Xg.GetValue(i));
        }
    } else if (mode == 3u) {
        // --- variant 3: block 0 only 串行所有 idx (确认 API 本身可用) ---
        if (GetBlockIdx() == 0u) {
            for (uint64_t i = 0ull; i < (uint64_t)N; ++i) {
                Yg.SetValue(i, Xg.GetValue(i));
            }
        }
    }
}
