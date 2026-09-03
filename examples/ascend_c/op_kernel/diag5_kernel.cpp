// Diag5 v3: 直接把 (CBIG, CCUB, CONE, sPOS) 存 Yg 看 LocalTensor 常数读得对不对.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void diag5_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; (void)x;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, 16ull);

    LocalTensor<float> sCBIG; sCBIG.SetSize(1); sCBIG.SetValue(0, 1.5957691216057308f);
    LocalTensor<float> sCCUB; sCCUB.SetSize(1); sCCUB.SetValue(0, 0.044715f);
    LocalTensor<float> sCONE; sCONE.SetSize(1); sCONE.SetValue(0, 1.0f);
    LocalTensor<float> sM1  ; sM1  .SetSize(1); sM1  .SetValue(0, -1.0f);
    LocalTensor<float> sM3  ; sM3  .SetSize(1); sM3  .SetValue(0, -3.5f);
    LocalTensor<float> sZERO; sZERO.SetSize(1); sZERO.SetValue(0, 0.0f);
    LocalTensor<float> sINF ; sINF .SetSize(1); sINF .SetValue(0, -1e20f);

    // 写入 Yg[0..7] 这些常数的 fp16 截断值.  fp16 范围够装所有值.
    Yg.SetValue(0, static_cast<half>(sCBIG.GetValue(0)));
    Yg.SetValue(1, static_cast<half>(sCCUB.GetValue(0)));
    Yg.SetValue(2, static_cast<half>(sCONE.GetValue(0)));
    Yg.SetValue(3, static_cast<half>(sM1  .GetValue(0)));
    Yg.SetValue(4, static_cast<half>(sM3  .GetValue(0)));
    Yg.SetValue(5, static_cast<half>(sZERO.GetValue(0)));
    Yg.SetValue(6, static_cast<half>(sINF .GetValue(0)));
    LocalTensor<float> sNFloat; sNFloat.SetSize(1); sNFloat.SetValue(0, 8.0f);
    Yg.SetValue(7, static_cast<half>(sNFloat.GetValue(0)));
}
