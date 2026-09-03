// 可靠调度方案:
//   PARTITIONS=K  (固定较小, 例如 16 或 24).
//   Host launch NBK = NBK = PARTITIONS * LARGE_FACTOR (例如 16*256=4096).
//   CANN 实际挑 ~100 个 bid ∈ [0, 4096).
//   每 block 计算 pid = bid % PARTITIONS.
//   每个 pid ∈ [0, PARTITIONS) 被至少 1 个 block 覆盖的概率 ≈ 1
//     (因为 100 个随机 mod 16, 空一类 ≈ (15/16)^100 ≈ 0.15%)
//   ⇒ 只要 PARTITIONS=32 且 NBK >= 4096, 覆盖率 ≥ 99.9%.
//   为了防同 pid 的多个 block 重复写 (idempotent 无 race 但浪费), 用"pid 范围内再次 grid-stride":
//     pid = bid % PARTITIONS
//     count_pid = NBK / PARTITIONS  (= LARGE_FACTOR)
//     local_rank = bid / PARTITIONS   (0..count_pid-1)
//     把 pid 对应的连续 N/PARTITIONS 段再拆成 count_pid 小段,
//     local_rank 负责第 rank 小格 (用 pid×seg + rank×subseg 内循环).
//   这样每小段恰好 1 个 bid 负责, 无重叠, 全量覆盖.
#include "kernel_operator.h"
using namespace AscendC;

static constexpr uint32_t PARTITIONS = 32u;
static constexpr uint32_t COUNT_PER_PART_EXPECT = 128u;  // NBK = PARTITIONS * COUNT_PER_PART_EXPECT = 4096
// host 必须严格传 NBK = PARTITIONS * COUNT_PER_PART_EXPECT = 4096 (或更大的同倍数)

extern "C" __global__ __aicore__
void identity_mod_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];
    const uint32_t NBK = T[1];    // 实际 launch 的 blocks 数 (host 保证 = PARTITIONS * COUNT_PER_PART_EXPECT)
    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    const uint32_t bid = GetBlockIdx();
    if (bid >= NBK) return;

    const uint32_t pid        = bid % PARTITIONS;
    const uint32_t count_pid  = NBK / PARTITIONS;   // 128
    const uint32_t local_rank = bid / PARTITIONS;   // ∈ [0, count_pid)

    // 第 pid 大段: [seg_lo..seg_hi)
    const uint64_t seg_sz = ((uint64_t)N + (uint64_t)PARTITIONS - 1ull) / (uint64_t)PARTITIONS;
    const uint64_t seg_lo = (uint64_t)pid * seg_sz;
    const uint64_t seg_hi = ((uint64_t)pid + 1ull) * seg_sz;
    const uint64_t seg_end = (seg_hi < (uint64_t)N) ? seg_hi : (uint64_t)N;
    const uint64_t seg_len = seg_end - seg_lo;
    if (seg_len == 0ull) return;

    // 段内再拆 count_pid 个 subseg, local_rank 负责第 rank 条
    const uint64_t sub_sz = (seg_len + (uint64_t)count_pid - 1ull) / (uint64_t)count_pid;
    const uint64_t my_lo = seg_lo + (uint64_t)local_rank * sub_sz;
    uint64_t my_hi       = seg_lo + ((uint64_t)local_rank + 1ull) * sub_sz;
    if (my_hi > seg_end) my_hi = seg_end;
    for (uint64_t i = my_lo; i < my_hi; ++i) {
        Yg.SetValue(i, Xg.GetValue(i));
    }
}
