// =============================================================================
// RoPE kernel — Ascend C (CANN 原生)
//
// 交错配对 (interleaved, RoFormer 原版) 旋转位置编码:
//   对每个 token t, 把 q/k 向量拆成 D/2 对相邻分量:
//     pair_a = (x[2a], x[2a+1])
//   用预计算好的 cos/sin 表 (t 行, D/2 列, host 侧算好, docs/04 §5.2) 旋转:
//     y[2a]   = x1·cos - x2·sin
//     y[2a+1] = x1·sin + x2·cos
//
// 纯逐元素乘加 (Vector 的主场); 教学版采用逐元素标量实现 (与 softmax_kernel
// 同规范)。cos/sin 表与 q/k 同为 fp16, kernel 内升 fp32 做乘加。
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void rope_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR cos_table, GM_ADDR sin_table,
                 GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];   // token 数
    const uint32_t D        = T[1];   // head dim (偶数)
    const uint32_t HALF     = D / 2u;

    GlobalTensor<half> Qg;
    Qg.SetGlobalBuffer((__gm__ half*)q, uint64_t(num_rows) * D);
    GlobalTensor<half> Kg;
    Kg.SetGlobalBuffer((__gm__ half*)k, uint64_t(num_rows) * D);
    GlobalTensor<half> Cg;
    Cg.SetGlobalBuffer((__gm__ half*)cos_table, uint64_t(num_rows) * HALF);
    GlobalTensor<half> Sg;
    Sg.SetGlobalBuffer((__gm__ half*)sin_table, uint64_t(num_rows) * HALF);
    GlobalTensor<half> QOg;
    QOg.SetGlobalBuffer((__gm__ half*)q_out, uint64_t(num_rows) * D);
    GlobalTensor<half> KOG;
    KOG.SetGlobalBuffer((__gm__ half*)k_out, uint64_t(num_rows) * D);

    for (uint64_t t = 0ull; t < (uint64_t)num_rows; ++t) {
        const uint64_t base  = t * (uint64_t)D;
        const uint64_t cbase = t * (uint64_t)HALF;
        for (uint64_t a = 0ull; a < (uint64_t)HALF; ++a) {
            // fp16 → fp32, 复数乘 (旋转)
            const float x1 = static_cast<float>(Qg.GetValue(base + 2ull * a));
            const float x2 = static_cast<float>(Qg.GetValue(base + 2ull * a + 1ull));
            const float k1 = static_cast<float>(Kg.GetValue(base + 2ull * a));
            const float k2 = static_cast<float>(Kg.GetValue(base + 2ull * a + 1ull));
            const float c  = static_cast<float>(Cg.GetValue(cbase + a));
            const float s  = static_cast<float>(Sg.GetValue(cbase + a));
            QOg.SetValue(base + 2ull * a,     static_cast<half>(x1 * c - x2 * s));
            QOg.SetValue(base + 2ull * a + 1ull, static_cast<half>(x1 * s + x2 * c));
            KOG.SetValue(base + 2ull * a,     static_cast<half>(k1 * c - k2 * s));
            KOG.SetValue(base + 2ull * a + 1ull, static_cast<half>(k1 * s + k2 * c));
        }
    }
}
