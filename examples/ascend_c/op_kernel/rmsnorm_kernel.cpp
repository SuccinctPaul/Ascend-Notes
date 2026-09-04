// =============================================================================
// RMSNorm kernel — Ascend C (CANN 原生)
//
// 按行计算 Root Mean Square Layer Normalization:
//   rms      = sqrt( (1/D) · Σ_j x_j² + eps )
//   y[j]     = (x[j] / rms) · gamma[j]
//
// 采用逐元素标量实现 (与 softmax_kernel 同规范)。2 pass:
//   Pass 1: fp32 累加 Σx² (归约必须宽精度, 否则长行精度崩)
//   Pass 2: inv_rms = 1/sqrt(Σx²/D + eps) (每行只算一次)
//           y = x · inv_rms · gamma (乘倒数, 不做逐元素除)
//
// 实现注意 (CANN 9.0 实测踩坑, 见 docs/04-ascend-c.md):
//   - tiling 常数用 GlobalTensor<float>.GetValue 标量读, 不走
//     "DataCopy 到裸 LocalTensor" —— 后者在部分 kernel 里会被优化丢;
//   - 标量工作张量 (sqrt 输入/输出) 用 TPipe/TBuf 分配**真实 UB**,
//     裸 LocalTensor<T>{} + SetSize 无后备存储, Sqrt 结果不可靠;
//   - aicore 内不允许整型变量→浮点 cast, 1/D 由 host 算好经 tiling 下发。
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void rmsnorm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];
    const uint32_t D        = T[1];
    // cf[8] at tiling offset 16 (bytes): cf[0]=eps, cf[1]=0.0, cf[2]=1.0, cf[3]=D 的浮点值
    GlobalTensor<float> Cf;
    Cf.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    const float EPS   = Cf.GetValue(0);
    const float ZERO  = Cf.GetValue(1);
    const float CONE  = Cf.GetValue(2);
    const float DF    = Cf.GetValue(3);

    GlobalTensor<half> Xg;
    Xg.SetGlobalBuffer((__gm__ half*)x, uint64_t(num_rows) * D);
    GlobalTensor<half> Gg;
    Gg.SetGlobalBuffer((__gm__ half*)gamma, uint64_t(D));
    GlobalTensor<half> Yg;
    Yg.SetGlobalBuffer((__gm__ half*)y, uint64_t(num_rows) * D);

    // ---- 标量工作张量: TPipe 分配真实 UB (各 32B 对齐一个槽) ----
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufIn, bufOut;
    pipe.InitBuffer(bufIn, 32);
    pipe.InitBuffer(bufOut, 32);
    LocalTensor<float> sSQ  = bufIn.Get<float>(1);    // sqrt 输入: Σx²/D + eps
    LocalTensor<float> sRMS = bufOut.Get<float>(1);   // sqrt 输出: rms

    for (uint64_t row = 0ull; row < (uint64_t)num_rows; ++row) {
        const uint64_t base = row * (uint64_t)D;
        // ---- Pass 1: fp32 累加 Σx² ----
        float sq_sum = ZERO;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            const float xv = static_cast<float>(Xg.GetValue(base + c));
            sq_sum += xv * xv;
        }
        // ---- inv_rms = 1 / sqrt(Σx²/D + eps) ----
        sSQ.SetValue(0, sq_sum / DF + EPS);
        Sqrt(sRMS, sSQ, 1);
        const float inv_rms = CONE / sRMS.GetValue(0);
        // ---- Pass 2: y = x · inv_rms · gamma (乘倒数, 不做逐元素除) ----
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            const float xv = static_cast<float>(Xg.GetValue(base + c));
            const float gv = static_cast<float>(Gg.GetValue(c));
            Yg.SetValue(base + c, static_cast<half>(xv * inv_rms * gv));
        }
    }
}
