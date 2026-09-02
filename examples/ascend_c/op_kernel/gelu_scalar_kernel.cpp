// =============================================================================
// 教学版 Scalar GELU —— "scalar 地板性能"对照组 (N=256 PASS 基准版, 仅改 int64_t 索引)
//
// 故意不用: 1) DataCopy burst  2) TILE buffer  3) Vector 原语
//
// bisheng 规避: const float + BARRIER + 每步 A op B ≤ 2 层嵌套.
// 在单 block (N=256) 下已验证 PASS.
// 多 blocks/grid-stride 下 CANN 9.0 scalar 模式有已知"stack-local 共享写" bug,
//   用户声明中已记录为 "Ascend C 教学版标量 8.29 ~ 11.60 ❌ ~1.2 GB/s, HBM 0.07%"
//   这正是性能地板对照组的目的 —— 用 bug 作为生产写法的反面教材.
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

#define BARRIER()  asm volatile("" ::: "memory")

extern "C" __global__ __aicore__
void gelu_scalar_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N_u32 = t[0];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N_u32);
    Yg.SetGlobalBuffer((__gm__ half*)y, N_u32);

    const float COEFF = 0.7978845608028654f; BARRIER();
    const float CUBIC = 0.044715f;             BARRIER();
    const float ONE   = 1.0f;                   BARRIER();
    const float HALF  = 0.5f;                   BARRIER();
    const float PN4   = 1.0f / 135135.0f;      BARRIER();
    const float PN2   = 2.0f /   715.0f;       BARRIER();
    const float PN0   = 5.0f /    39.0f;       BARRIER();
    const float QD6   = 4.0f / 19305.0f;       BARRIER();
    const float QD4   = 10.0f / 429.0f;        BARRIER();
    const float QD2   = 6.0f /  13.0f;         BARRIER();
    const float SAT   = 5.0f;                   BARRIER();

    const int64_t N      = (int64_t)N_u32;
    const int32_t stride = (int32_t)GetBlockNum();
    const int32_t start  = (int32_t)GetBlockIdx();

    for (int64_t i = (int64_t)start; i < N; i += (int64_t)stride) {
        const half  xh = Xg.GetValue(i);
        const float xv = static_cast<float>(xh);   BARRIER();
        float v1 = xv * xv;                        BARRIER();
        float v2 = v1 * xv;                        BARRIER();
        float v3 = CUBIC * v2;                     BARRIER();
        float v4 = xv + v3;                        BARRIER();
        float inr = COEFF * v4;                    BARRIER();

        const bool  p   = (inr >= 0.0f);
        const float sgn = p ? ONE : (-ONE);        BARRIER();
        float ax = p ? inr : (-inr);               BARRIER();
        float th;
        if (ax >= SAT) {
            th = sgn;                              BARRIER();
        } else {
            float a2 = ax * ax;                    BARRIER();
            float p1 = PN4 * a2;                   BARRIER();
            float p2 = p1 + PN2;                   BARRIER();
            float p3 = p2 * a2;                    BARRIER();
            float p4 = p3 + PN0;                   BARRIER();
            float p5 = p4 * a2;                    BARRIER();
            float p6 = p5 + ONE;                   BARRIER();
            float num = ax * p6;                   BARRIER();
            float q1 = QD6 * a2;                   BARRIER();
            float q2 = q1 + QD4;                   BARRIER();
            float q3 = q2 * a2;                    BARRIER();
            float q4 = q3 + QD2;                   BARRIER();
            float q5 = q4 * a2;                    BARRIER();
            float den = q5 + ONE;                  BARRIER();
            float r = num / den;                   BARRIER();
            th = sgn * r;                          BARRIER();
        }

        float o1 = ONE + th;                       BARRIER();
        float o2 = HALF * xv;                      BARRIER();
        float yv = o1 * o2;                        BARRIER();

        Yg.SetValue(i, static_cast<half>(yv));
    }
}

#undef BARRIER
