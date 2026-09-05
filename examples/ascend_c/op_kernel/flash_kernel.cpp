// =============================================================================
// FlashAttention 前向 kernel — Ascend C (CANN 原生, 逐行 online softmax 教学版)
//
// 对应 docs/ops/07-flash-attention.md: Flash 的算法本质 = 分块 + online softmax
// (m/l/acc 增量, L×S 分数矩阵不整体物化)。本教学版对每个 query 行 (H×L 展开):
//   Pass 1: 逐 s 打分求行 max (running max, 分数暂存 scratch)
//   Pass 2: 逐 s exp(score-mx) → 累加 l → 加权累加 acc (借助 OUT 缓冲原位累加)
//   Pass 3: out = acc / l 归一化写回
// online 语义与 triton 版一致, 差别仅在串行标量 (教学地板, 无分块并行)。
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void flash_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR scratch,
                  GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t H  = T[0];
    const uint32_t L  = T[1];
    const uint32_t S  = T[2];
    const uint32_t D  = T[3];
    GlobalTensor<float> Cf;
    Cf.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    const float INV_SQRT_D = Cf.GetValue(0);

    const uint32_t R = H * L;
    GlobalTensor<half> Qg;
    Qg.SetGlobalBuffer((__gm__ half*)q, uint64_t(R) * D);
    GlobalTensor<half> Kg;
    Kg.SetGlobalBuffer((__gm__ half*)k, uint64_t(H) * S * D);
    GlobalTensor<half> Vg;
    Vg.SetGlobalBuffer((__gm__ half*)v, uint64_t(H) * S * D);
    GlobalTensor<half> OUTg;
    OUTg.SetGlobalBuffer((__gm__ half*)out, uint64_t(R) * D);
    GlobalTensor<half> SCg;   // scratch: Pass1 分数 / Pass2 p 值 (R*S, fp16)
    SCg.SetGlobalBuffer((__gm__ half*)scratch, uint64_t(R) * S);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufIn, bufOut;
    pipe.InitBuffer(bufIn, 32);
    pipe.InitBuffer(bufOut, 32);
    LocalTensor<float> sIN  = bufIn.Get<float>(1);
    LocalTensor<float> sEXP = bufOut.Get<float>(1);

    for (uint32_t r = 0u; r < R; ++r) {
        const uint32_t h = r / L;
        const uint64_t qbase = uint64_t(r) * D;
        // ---- Pass 1: 打分 (暂存 scratch) + running max ----
        float m = -1e30f;
        for (uint32_t s = 0u; s < S; ++s) {
            const uint64_t kbase = (uint64_t(h) * S + s) * D;
            float sc = 0.0f;
            for (uint32_t d = 0u; d < D; ++d) {
                sc += static_cast<float>(Qg.GetValue(qbase + d)) *
                      static_cast<float>(Kg.GetValue(kbase + d));
            }
            sc *= INV_SQRT_D;
            if (sc > m) m = sc;
            SCg.SetValue(uint64_t(r) * S + s, static_cast<half>(sc));
        }
        // ---- Pass 2: p = exp(score - m) 暂存; acc 借 OUT 缓冲原位累加 ----
        float l = 0.0f;
        for (uint32_t s = 0u; s < S; ++s) {
            sIN.SetValue(0, static_cast<float>(SCg.GetValue(uint64_t(r) * S + s)) - m);
            Exp(sEXP, sIN, 1);
            const float p = sEXP.GetValue(0);
            SCg.SetValue(uint64_t(r) * S + s, static_cast<half>(p));
            l += p;
            const uint64_t vbase = (uint64_t(h) * S + s) * D;
            for (uint32_t d = 0u; d < D; ++d) {
                const float acc = static_cast<float>(OUTg.GetValue(qbase + d)) +
                                  p * static_cast<float>(Vg.GetValue(vbase + d));
                OUTg.SetValue(qbase + d, static_cast<half>(acc));
            }
        }
        // ---- Pass 3: 归一化 ----
        const float inv_l = 1.0f / l;
        for (uint32_t d = 0u; d < D; ++d) {
            OUTg.SetValue(qbase + d, static_cast<half>(static_cast<float>(OUTg.GetValue(qbase + d)) * inv_l));
        }
    }
}
