// =============================================================================
// Softmax 生产版 kernel — Ascend C (CANN 原生)
//
// 按行计算 numerically stable softmax:
//   y[i] = exp(x[i] - rowmax) / Σ_j exp(x[j] - rowmax)
//
// 采用逐元素标量实现 (与 GELU v6 同规范), 避免 Vector tile 在 CANN 9.0 教学
// 环境下的别名/未初始化错误.  3 pass: row_max / exp+sum / normalize.
//
// 常数 INF=非常小的负值 / ZERO / ONE / INV 都从 tiling GM DataCopy 取,
// 规避 CANN 9.0 "LocalTensor SetValue(立即数) + GetValue → 读到 -inf" 的 bug.
// Host 下发 numBlocks=1 规避 CANN 9 多 block grid-stride bid 随机调度遗漏.
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void softmax_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];
    const uint32_t D        = T[1];
    // T[2] = unused_pad (uint32)
    // cf[8] at T+4 (bytes 16..48):
    //   cf[0]=-1e20 (M_INF), cf[1]=0.0 (ZERO), cf[2]=1.0 (CONE)
    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);
    const float M_INF = Cl.GetValue(0);
    const float ZERO  = Cl.GetValue(1);
    const float CONE  = Cl.GetValue(2);

    GlobalTensor<half> Xg;
    Xg.SetGlobalBuffer((__gm__ half*)x, uint64_t(num_rows) * D);
    GlobalTensor<half> Yg;
    Yg.SetGlobalBuffer((__gm__ half*)y, uint64_t(num_rows) * D);

    LocalTensor<float> sXV ; sXV .SetSize(1);
    LocalTensor<float> sSH ; sSH .SetSize(1);
    LocalTensor<float> sEXP; sEXP.SetSize(1);
    LocalTensor<float> sINV; sINV.SetSize(1);

    for (uint64_t row = 0ull; row < (uint64_t)num_rows; ++row) {
        const uint64_t base = row * (uint64_t)D;
        // ---- Pass 1: row max ----
        float row_max = M_INF;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sXV.SetValue(0, static_cast<float>(Xg.GetValue(base + c)));
            const float xv = sXV.GetValue(0);
            if (xv > row_max) row_max = xv;
        }
        // ---- Pass 2: exp(x-row_max) accumulate sum, write Yg temporarily ----
        float sum_e = ZERO;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sXV.SetValue(0, static_cast<float>(Xg.GetValue(base + c)));
            // 同构已验证过的 softmax 写法: sSH = sXV.GetValue(0) - row_max
            sSH.SetValue(0, sXV.GetValue(0) - row_max);
            Exp(sEXP, sSH, 1);
            const float ev = sEXP.GetValue(0);
            sum_e += ev;
            Yg.SetValue(base + c, static_cast<half>(ev));
        }
        // ---- Pass 3: y_i = exp_i / sum_e  (乘倒数, 因 div 逐元素比 fp32 div 快) ----
        sINV.SetValue(0, CONE / sum_e);
        const float inv = sINV.GetValue(0);
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sEXP.SetValue(0, static_cast<float>(Yg.GetValue(base + c)));
            const float yv = sEXP.GetValue(0) * inv;
            Yg.SetValue(base + c, static_cast<half>(yv));
        }
    }
}
