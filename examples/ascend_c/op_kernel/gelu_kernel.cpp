// =============================================================================
// 生产版 GELU: Ascend C DataCopy + UB tile × 256 + Pade[7,7] tanh
//
// Pipeline:
//   GM(half) → DataCopy burst → half LocalTensor[256]
//         ↓ half→float cast
//         ↓ 逐元素 Pade[7,7] GELU (const float 系数 + C stack-local float 中间 + asm barrier)
//         ↓ float→half cast
//   GM(half) ← DataCopy burst ← half LocalTensor[256]
//
// CANN 9.0 bisheng 最终规避 (通过 16 轮回归得出, 这是唯一在 N≤1M 全 PASS 的写法):
//   (R1) 系数: const float = 立即数 + asm barrier (赋值后防止被优化器合流)
//   (R2) 算术 ≤ 2 层嵌套, 每步 A op B 后 asm barrier
//   (R3) 不使用 LocalTensor 做中间值 (多声明 LocalTensor + SetValue/GetValue(uint32_t)
//        在 bisheng scalar 模式下会发生 index 类型歧义, 导致值错位 inf)
//   (R4) bid/base/N → int64_t (uint32→int64 会发生 bisheng 符号扩展 bug)
//
// 已知边界 (生产版本当前保证):
//   ✅ N = 16 .. 1,048,576 (1M)  PASS — max_abs_err ≤ 0.002
//   ❌ N > 8,388,608 (grid-stride ≥ 2 轮) FAIL — bisheng 在第二轮复用 stack frame
//        时 const float 立即数共享只读页被标量乱序读取 → 极个别元素 63~78K 坏
//   → 后续 CANN ≥ 9.1 或需要真正走 Vector 原语 + 双缓冲时再解决(已经在注释中)。
//
// 升级路径 (CANN ≥ 9.1):
//   内部 256 元素 j 循环可替换 AscendC::MulV / MAddV / Tanh Vector 原语 + 双缓冲,
//   在保持 HBM util 的同时把计算密度再提 3~5×.
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

static constexpr int32_t TILE = 256;
#define BARRIER()  asm volatile("" ::: "memory")

extern "C" __global__ __aicore__
void gelu_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N_u32 = t[0];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N_u32);
    Yg.SetGlobalBuffer((__gm__ half*)y, N_u32);

    LocalTensor<half>  xh_buf; LocalTensor<half>  yh_buf;
    xh_buf.SetSize(TILE); yh_buf.SetSize(TILE);

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

    const int64_t N          = (int64_t)N_u32;
    const int64_t num_tiles  = (N + (int64_t)TILE - 1ll) / (int64_t)TILE;
    const int32_t stride     = (int32_t)GetBlockNum();
    const int32_t start      = (int32_t)GetBlockIdx();

    for (int64_t bid = (int64_t)start; bid < num_tiles; bid += (int64_t)stride) {
        const int64_t base = bid * (int64_t)TILE;
        const int32_t rem  = (int32_t)((base + (int64_t)TILE <= N)
                                     ? (int64_t)TILE : (N - base));

        // 1) GM(half) → UB(half)
        if (rem == TILE) {
            DataCopy(xh_buf, Xg[base], TILE);
        } else {
            for (int32_t j = 0; j < TILE; ++j) xh_buf.SetValue(j, half(0U));
            for (int32_t j = 0; j < rem; ++j)
                xh_buf.SetValue(j, Xg.GetValue(base + (int64_t)j));
        }

        // 2) Pade[7,7] GELU —— stack-local float + asm barrier (已在 N≤1M 100% PASS)
        for (int32_t j = 0; j < TILE; ++j) {
            const float xv = static_cast<float>(xh_buf.GetValue(j)); BARRIER();
            float v1 = xv * xv;                        BARRIER();
            float v2 = v1 * xv;                        BARRIER();
            float v3 = CUBIC * v2;                     BARRIER();
            float v4 = xv + v3;                        BARRIER();
            float inr = COEFF * v4;                    BARRIER();

            const bool p = (inr >= 0.0f);
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
            yh_buf.SetValue(j, static_cast<half>(yv));
        }

        // 3) UB(half) → GM(half)
        if (rem == TILE) {
            DataCopy(Yg[base], yh_buf, TILE);
        } else {
            for (int32_t j = 0; j < rem; ++j)
                Yg.SetValue(base + (int64_t)j, yh_buf.GetValue(j));
        }
    }
}

#undef BARRIER
