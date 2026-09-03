// 每个 block 通过 atomicAdd(workspace[0]) 抢一个连续实例号 instance_id ∈ [0, actual-1]
// 实际同时运行的 blocks 总数 actual = 最后抢到的值 + 1 (实际就是 ~94 动态)
// stride=actual, 但我们无法在 kernel 内同步获知 actual.
// ⇒ 折中: 预分配 ACTUAL_CAP=256 个实例槽, stride=ACTUAL_CAP
//         block 抢 id ∈ [0, ACTUAL_CAP), 抢不到 (id>=ACTUAL_CAP) 直接 return.
//         只要 ACTUAL_CAP > 实际 ~94 blocks, stride=256 且 [0..actual-1] 都有人跑
//         ⇒ 每个 i%256 ∈ [0, actual-1] 被覆盖, 剩下 actual..255 的余数类没人跑!
// 解决: 改成"抢 N 的分块": 把 N 分成 N_CHUNKS=2048 小块, 每小块 ceil(N/N_CHUNKS) 连续 idx.
//       每个 block 抢一个 chunk_id (0..N_CHUNKS-1), 处理 idx = [chunk_id*CHUNK_SZ .. min(N,(chunk_id+1)*CHUNK_SZ))
//       只要实际 blocks (~94) × 循环抢 = 最终抢到 2048 个, 全部覆盖. (一个 block 可以抢多个)
#include "kernel_operator.h"
using namespace AscendC;

static constexpr uint32_t N_CHUNKS = 4096u;   // 分块数 > 实际 blocks 数 × 4

extern "C" __global__ __aicore__
void identity_chunk_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // workspace[0..1] = uint32 atomic counter (初始 0), 然后 N_CHUNKS=4096
    __gm__ uint32_t* COUNTER = reinterpret_cast<__gm__ uint32_t*>(workspace);

    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];
    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    const uint32_t CHUNK_SZ = (N + N_CHUNKS - 1u) / N_CHUNKS;  // 最后一块小
    if (CHUNK_SZ == 0u) return;

    while (true) {
        // 原子加 1, 获取 chunk_id.  若 >= N_CHUNKS 就退出.
        const uint32_t chunk_id = AtomicAdd(COUNTER, 1u);
        if (chunk_id >= N_CHUNKS) break;

        const uint64_t lo = (uint64_t)chunk_id * CHUNK_SZ;
        const uint64_t hi = (uint64_t)(chunk_id + 1u) * CHUNK_SZ;
        const uint64_t end = (hi < (uint64_t)N) ? hi : (uint64_t)N;
        for (uint64_t i = lo; i < end; ++i) {
            Yg.SetValue(i, Xg.GetValue(i));
        }
    }
}
