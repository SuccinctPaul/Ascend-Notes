// Diag6: 验证 DataCopy(LocalTensor<float>, GlobalTensor<float>, count) 是否正常复制 tiling 常数.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void diag6_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; (void)x;
    // tiling:  8 bytes u32 padding, 然后 8 floats (cf[0..7])
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    GlobalTensor<float> Cg;
    Cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 2), 8u);
    LocalTensor<float> Cl; Cl.SetSize(8);
    DataCopy(Cl, Cg, 8u);

    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, 16ull);
    // 写每个 float(Cl[0..7]) -> half(y[i]). 如果 DataCopy 工作了 Yg[0] = fp16(1.5958) = 1.5957
    for (uint32_t i = 0u; i < 8u; ++i) {
        Yg.SetValue(i, static_cast<half>(Cl.GetValue(i)));
    }
    // 作为控制组: 把同样几个 float 通过 (GlobalTensor half GetValue → LocalTensor half 复制)
    // 不, 作为控制组我们直接 SetValue(立即数) 的 CONST LocalTensor 输出 - 已知是 -inf.
    LocalTensor<float> s1; s1.SetSize(1); s1.SetValue(0, 1.5957691216057308f);
    Yg.SetValue(8, static_cast<half>(s1.GetValue(0)));
    // 控制 2: 通过在循环里赋值确保变量"被用到"
    LocalTensor<float> acc; acc.SetSize(1); acc.SetValue(0, 0.0f);
    for (uint32_t k = 0u; k < 8u; ++k) {
        LocalTensor<float> tmp; tmp.SetSize(1);
        tmp.SetValue(0, Cl.GetValue(k));
        acc.SetValue(0, acc.GetValue(0) + tmp.GetValue(0));
    }
    Yg.SetValue(9, static_cast<half>(acc.GetValue(0)));
}
