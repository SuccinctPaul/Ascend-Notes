// =============================================================================
// GELU 生产版 kernel — Ascend C (CANN 原生)
//
// 实现公式 (EXP 等价形式, 与 tanh 近似数学完全等价):
//   GELU(x) = x / (1 + exp(-CBIG * (x + CCUB * x^3)))
//   其中 CBIG = 2*sqrt(2/pi) ≈ 1.5958,  CCUB = 0.044715.
//
// CANN 9.0 已知 bug 规避策略 (2025-04-17, 容器环境 910B2):
//   1) 禁止用 LocalTensor<float>.SetValue(idx, 编译期 float 立即数) 再 GetValue 当常数.
//      AICore 编译器会把这些值当成 UB 未初始化读出 (-inf). 常数必须从 tiling GM
//      通过 DataCopy(PIPE_MTE2) 搬到 LocalTensor<float> 再 GetValue.
//   2) 负数构造必须是 "两个数据流来源 fp32 变量的减法", 左操作数写成
//      `sXV.GetValue(0)` (mirror 已验证通过的 softmax 写法), 右操作数是 local fp32.
//   3) CANN 调度器在多 block 下只会执行 ~90/任意个 bid, 不能用 grid-stride.
//      host 强制传 numBlocks=1.
//
// 核内循环逐元素标量, 不使用 Vector tile (DataCopy/Muls/Adds 等), 因为教学环境
// Vector tile 的 256B slot 数据搬运会在 CANN 9.0 下返回未初始化/别名错误.
// 与 scalar 版保持同一内循环, 但 scalar 版增加额外指令延迟模拟地板性能.
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];

    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 2), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);
    const float CBIG = Cl.GetValue(0);
    const float CCUB = Cl.GetValue(1);
    const float CONE = Cl.GetValue(2);

    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<float> sXV ; sXV .SetSize(1);
    LocalTensor<float> sSH ; sSH .SetSize(1);
    LocalTensor<float> sEXP; sEXP.SetSize(1);

    for (uint64_t i = 0ull; i < (uint64_t)N; ++i) {
        sXV.SetValue(0, static_cast<float>(Xg.GetValue(i)));
        const float xv = sXV.GetValue(0);
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float pos = CBIG * t1;
        const float big = xv + pos;
        // 同构 softmax sSH.SetValue(0, sXV.GetValue(0) - row_max)
        sSH.SetValue(0, sXV.GetValue(0) - big);
        Exp(sEXP, sSH, 1);
        const float den = CONE + sEXP.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
