// GELU v5 kernel: 常数从 tiling GM 通过 DataCopy 搬到 UB LocalTensor, 然后走 LocalTensor.GetValue.
// Tiling 布局 (严格 Ascend C DataCopy 要求):
//   offset 0:   uint32_t N           (4 bytes)
//   offset 4:   uint32_t padding     (4 bytes, 让 float[] 8-byte 对齐)
//   offset 8:   float constants[3]   (12 bytes) = {CBIG=2*sqrt(2/pi), CCUB=0.044715, CONE=1.0}
// 注: Ascend C DataCopy(dst, src, count) 要求 dst/src/burst/count 对应长度匹配。
//     对 float32 的 GlobalTensor 与 LocalTensor, 3 个元素 * 4B = 12B ≤ 32B (单 burst 64B?
//     Ascend C 文档对 float DataCopy 允许 burstLen<=32B? 先用 burst=1, count=32B (8 floats) 足够覆盖 3 floats.
// 安全起见用 8 floats 的 padding.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gelu_v5_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // tiling: [uint32 N, uint32 pad, float CBIG, float CCUB, float CONE]
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = T[0];
    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 2), 8u);  // 从偏移 8B 开始读 8 个 float
    LocalTensor<float> Cl; Cl.SetSize(8);
    // 8 floats = 32 bytes.  用单-count (元素个数) 三参 DataCopy.
    DataCopy(Cl, Cg, 8u);

    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<float> sARG; sARG.SetSize(1);
    LocalTensor<float> sEXP; sEXP.SetSize(1);
    LocalTensor<float> sXV ; sXV .SetSize(1);
    LocalTensor<float> sBIG; sBIG.SetSize(1);

    // constants 从 LocalTensor Cl 读 (已经是数据流, 非 SetValue(immediate))
    //   Cl[0] = CBIG, Cl[1] = CCUB, Cl[2] = CONE
    const float CBIG = Cl.GetValue(0);
    const float CCUB = Cl.GetValue(1);
    const float CONE = Cl.GetValue(2);

    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        sXV.SetValue(0, static_cast<float>(Xg.GetValue(i)));
        const float xv = sXV.GetValue(0);
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float pos = CBIG * t1;
        sBIG.SetValue(0, xv + pos);
        const float big = sBIG.GetValue(0);
        // 减法严格 mirror softmax: sXV.GetValue(0) - big
        sARG.SetValue(0, sXV.GetValue(0) - big);
        Exp(sEXP, sARG, 1);
        const float den = CONE + sEXP.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
