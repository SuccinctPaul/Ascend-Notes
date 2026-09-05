// =============================================================================
// GQA 解码注意力 kernel — Ascend C (CANN 原生, 标量教学版)
//
// 对应 docs/ops/06-gqa-kvcache.md: 解码一步 —
//   scores[hq,s] = ( q[hq] · K[kv,s] ) / sqrt(D),  kv = hq / G,  G = Hq/Hkv
//   p = softmax(scores) (数值稳定: 减 max)
//   out[hq,d]   = Σ_s p[hq,s] · V[kv,s,d]
//
// 3-pass 标量实现 (与 softmax_kernel 同规范):
//   Pass 1: 打分 → fp32 scratch (host 分配), 同时求行 max
//   Pass 2: exp(score-max) → scratch (fp16), 求和 l
//   Pass 3: 逐 d 加权累加 scratch 里的 p × V, 乘 1/l
// 常数经 tiling (GlobalTensor float GetValue); Exp 用 TPipe 真实 UB (同 rmsnorm)。
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gqa_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR scratch,
                GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t Hq  = T[0];
    const uint32_t G   = T[1];   // 每组的 query 头数 = Hq/Hkv (host 预计算)
    const uint32_t S   = T[2];
    const uint32_t D   = T[3];
    GlobalTensor<float> Cf;
    Cf.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    const float INV_SQRT_D = Cf.GetValue(0);

    GlobalTensor<half> Qg;
    Qg.SetGlobalBuffer((__gm__ half*)q, uint64_t(Hq) * D);
    GlobalTensor<half> Kg;
    Kg.SetGlobalBuffer((__gm__ half*)k, uint64_t(Hq / G) * S * D);
    GlobalTensor<half> Vg;
    Vg.SetGlobalBuffer((__gm__ half*)v, uint64_t(Hq / G) * S * D);
    GlobalTensor<half> OUTg;
    OUTg.SetGlobalBuffer((__gm__ half*)out, uint64_t(Hq) * D);
    GlobalTensor<half> SCg;   // scratch: 分数/exp 值 (Hq*S, fp16)
    SCg.SetGlobalBuffer((__gm__ half*)scratch, uint64_t(Hq) * S);

    // Exp 工作张量: TPipe 真实 UB (同 rmsnorm Sqrt 的写法)
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufIn, bufOut;
    pipe.InitBuffer(bufIn, 32);
    pipe.InitBuffer(bufOut, 32);
    LocalTensor<float> sIN  = bufIn.Get<float>(1);
    LocalTensor<float> sEXP = bufOut.Get<float>(1);

    for (uint32_t hq = 0u; hq < Hq; ++hq) {
        const uint32_t kv = hq / G;
        const uint64_t qbase = uint64_t(hq) * D;
        // ---- Pass 1: 打分 + 行 max ----
        float m = -1e30f;
        for (uint32_t s = 0u; s < S; ++s) {
            const uint64_t kbase = (uint64_t(kv) * S + s) * D;
            float sc = 0.0f;
            for (uint32_t d = 0u; d < D; ++d) {
                sc += static_cast<float>(Qg.GetValue(qbase + d)) *
                      static_cast<float>(Kg.GetValue(kbase + d));
            }
            sc *= INV_SQRT_D;
            if (sc > m) m = sc;
            SCg.SetValue(uint64_t(hq) * S + s, static_cast<half>(sc));
        }
        // ---- Pass 2: exp(score - m) → scratch, 求和 l ----
        float l = 0.0f;
        for (uint32_t s = 0u; s < S; ++s) {
            sIN.SetValue(0, static_cast<float>(SCg.GetValue(uint64_t(hq) * S + s)) - m);
            Exp(sEXP, sIN, 1);
            const float e = sEXP.GetValue(0);
            l += e;
            SCg.SetValue(uint64_t(hq) * S + s, static_cast<half>(e));
        }
        // ---- Pass 3: 逐 d 加权累加 × 1/l ----
        const float inv_l = 1.0f / l;
        for (uint32_t d = 0u; d < D; ++d) {
            float acc = 0.0f;
            for (uint32_t s = 0u; s < S; ++s) {
                const uint64_t vbase = (uint64_t(kv) * S + s) * D;
                acc += static_cast<float>(SCg.GetValue(uint64_t(hq) * S + s)) *
                       static_cast<float>(Vg.GetValue(vbase + d));
            }
            OUTg.SetValue(qbase + d, static_cast<half>(acc * inv_l));
        }
    }
}
