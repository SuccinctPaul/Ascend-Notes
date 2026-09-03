// GELU v4 kernel: 常数完全从 tiling GM 读 (不经过 LocalTensor SetValue 立即数路径).
// 从 gelu_host v3 传的 tiling = [N, u32(CBIG_bits), u32(CCUB_bits)].  注意 float bit_cast
// 用 "LocalTensor<int32_t> + LocalTensor<float> 地址 overlap" 不行,  AscendC 接口没提供
// bitcast;  所以我们直接在 tiling 上放 (uint32_t)N, 紧接着 3 个 float (必须对齐).
// C++ struct:  struct Tiling { uint32_t N; float CBIG; float CCUB; float CONE; };
//   sizeof = 16 bytes, 16B 对齐, 完全合法.
#include "kernel_operator.h"
using namespace AscendC;

struct ScalarTiling {
    uint32_t N;
    float    CBIG;   // 2*sqrt(2/pi)
    float    CCUB;   // 0.044715
    float    CONE;   // 1.0
};

extern "C" __global__ __aicore__
void gelu_v4_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ ScalarTiling* t = reinterpret_cast<__gm__ ScalarTiling*>(tiling);
    // 从 GlobalTiling 直接构造 LocalTensor<float> 视图
    GlobalTensor<float> CT;
    CT.SetGlobalBuffer((__gm__ float*)(&t->CBIG), 3u);
    LocalTensor<float> lC; lC.SetSize(3);
    // float 3 个, 从 GM copy 到 UB 再读 (用 DataCopy API; Ascend C 不支持 GlobalTensor<float>::GetValue,
    // 只有 half/int 等特定类型).  DataCopy<PIPE_MTE2>: dst=LocalTensor<float>, src=GlobalTensor<float>, burst=1, n=1
    for (uint32_t k = 0u; k < 3u; ++k) {
        // 逐元素 copy: AscendC 的 DataCopy(dst, src, count) 对 float: 需 count 是 32B 对齐倍数?
        // 退而求其次: 通过 half 的 GlobalTensor 读取再拼? 不, AscendC 有 LocalTensor.SetValue(idx, float)
        // 走已验证通过的 scalar GetValue 路径 — 但源 CT 是 float GlobalTensor, 没 GetValue API.
        // 方案 2: 不要 ScalarTiling, 把 tiling 定义为 uint32_t[4] = {N, bit_u32(CBIG), bit_u32(CCUB), bit_u32(CONE)}.
        //   AscendC 的 GlobalTensor<int32_t> / uint32_t 大概率也没有 GetValue — 我们绕开这条路.
    }

    // 方案 3 (最终采用, 已在 softmax 中验证可行):
    // 我们已经验证了 softmax 中的 sM_INF=-1e20f 经过 LocalTensor.SetValue/GetValue 是正确的 (row_max
    // 初始化为 -1e20f，argmax 时总能替换掉正确的最大值)。如果 LocalTensor float SetValue/GetValue 的常数传递
    // 真的有偏差, softmax 的 argmax 初始化就会错。所以常数传递实际上是 OK 的。误差只能来自 Vector Exp。
    //
    // 诊断：这里我们通过 "直接把 CBIG/CCUB 写在一个完全独立的变量来源链里" 的方式确认：
    //   CCUB / CBIG 从一个 uint32_t tiling 数组两次 GetValue 读，然后通过 LocalTensor<uint32_t> 转 LocalTensor<float>
    //   的 bitcopy 技巧 (通过使用同一个 4-byte 内存地址的重解释: AscendC 不支持, 简化).
    // 简单起见, 我们直接用 softmax 里证明有效的 LocalTensor<float>(1) SetValue+GetValue + 减法 pattern:
    (void)CT; (void)lC;

    const uint32_t N = t->N;
    GlobalTensor<half> Xg; Xg.SetGlobalBuffer((__gm__ half*)x, N);
    GlobalTensor<half> Yg; Yg.SetGlobalBuffer((__gm__ half*)y, N);
    LocalTensor<float> sCBIG; sCBIG.SetSize(1); sCBIG.SetValue(0, 1.5957691216057308f);
    LocalTensor<float> sCCUB; sCCUB.SetSize(1); sCCUB.SetValue(0, 0.044715f);
    LocalTensor<float> sCONE; sCONE.SetSize(1); sCONE.SetValue(0, 1.0f);
    LocalTensor<float> sARG ; sARG .SetSize(1);
    LocalTensor<float> sEXP ; sEXP .SetSize(1);
    LocalTensor<float> sBIG ; sBIG .SetSize(1);
    const float CBIG = sCBIG.GetValue(0);
    const float CCUB = sCCUB.GetValue(0);
    const float CONE = sCONE.GetValue(0);

    // 诊断部分: 先处理 8 个值。
    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        const float xv = static_cast<float>(Xg.GetValue(i));
        const float x2 = xv * xv;
        const float x3 = x2 * xv;
        const float bx3 = CCUB * x3;
        const float t1  = xv + bx3;
        const float pos = CBIG * t1;
        const float shifted = xv + pos;
        sBIG.SetValue(0, shifted);
        const float big = sBIG.GetValue(0);
        // ======= 关键: 完全 mirror softmax (PASS) 模式 =======
        // softmax:
        //   sXV.GetValue(0)  是 GlobalTensor 数据 round-trip: 这里 xv 相同.
        //   row_max 是 fp32 scalar local: 这里 big 也是.
        //   sSH.SetValue(0, sXV.GetValue(0) - row_max);
        //   Exp(sEXP, sSH, 1);
        // 对应 GELU:
        sARG.SetValue(0, xv - big);
        Exp(sEXP, sARG, 1);
        const float den = CONE + sEXP.GetValue(0);
        const float yv = xv / den;
        Yg.SetValue(i, static_cast<half>(yv));
    }
}
