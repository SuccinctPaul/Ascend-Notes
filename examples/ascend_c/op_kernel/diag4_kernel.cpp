// =============================================================================
// Diag4: 极简 fp32 单元 Vector 原语诊断 —— 每种原语单测, 定位到底谁坏了
// - 工作流: host 造已知的 256 个 float (全 1.0, 或 x 序列) → 填充到 half GM
//           kernel: half xh → scalar SetValue 转 xf float buffer
//                   运行一条 Vector 原语 (Adds/Mul/Muls/Exp/Div)
//                   float y_tmp → scalar 转 yh half → 写 GM
//           host 回读和预期比较.
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

static constexpr int32_t TILE = 256;

// testCase: 0=y=x+1(Adds); 1=y=x*x(Mul); 2=y=2*x(Muls); 3=y=exp(x)(Exp); 4=y=1/x(Div)
extern "C" __global__ __aicore__
void diag4_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N = t[0];
    const uint32_t num_tiles = (N + TILE - 1u) / TILE;

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<half>  xh_buf;
    LocalTensor<half>  yh_buf;
    LocalTensor<float> xf_buf;
    LocalTensor<float> tmp_buf;
    LocalTensor<float> tmp2_buf;
    xh_buf.SetSize(TILE); yh_buf.SetSize(TILE);
    xf_buf.SetSize(TILE); tmp_buf.SetSize(TILE); tmp2_buf.SetSize(TILE);

    const uint32_t stride = GetBlockNum();
    const uint32_t start  = GetBlockIdx();

    for (uint32_t bid = start; bid < num_tiles; bid += stride) {
        const uint32_t base = bid * TILE;
        const uint32_t rem  = (base + TILE <= N) ? TILE : (N - base);

        // ---- 1. scalar copy half xh → float xf (对齐无 DataCopy 的纯标量转义路径)
        // 这样避免任何 DataCopy 参数歧义. tail 填 gelu(0)=0
        for (int32_t j = 0; j < TILE; ++j) {
            if ((uint32_t)j < rem) {
                half hv = Xg.GetValue((uint64_t)base + j);
                xh_buf.SetValue(j, hv);
                xf_buf.SetValue(j, static_cast<float>(hv));
            } else {
                xh_buf.SetValue(j, half(0.0f));
                xf_buf.SetValue(j, 0.0f);
            }
        }

        // ---- 2. 执行 5 个 Vector 原语级联 (每个 tile 256, 全 fp32) ----
        //   y = x / (1 + exp(-2*1.595769*(x + 0.044715*x*x*x)))
        //   = full GELU (tanh via exp)
        const float alpha_f = 1.5957691216057308f;
        const float beta_f  = 0.044715f;
        const float neg1_f  = -1.0f;
        const float one_f   = 1.0f;

        // tmp_buf = x^2
        Mul(tmp_buf, xf_buf, xf_buf, TILE);
        // tmp_buf = x * x^2 = x^3
        Mul(tmp_buf, xf_buf, tmp_buf, TILE);
        // tmp_buf = beta * x^3
        Muls(tmp_buf, tmp_buf, beta_f, TILE);
        // tmp_buf = x + beta*x^3
        Add(tmp_buf, xf_buf, tmp_buf, TILE);
        // tmp_buf = alpha*(x + beta*x^3) = 2*inner
        Muls(tmp_buf, tmp_buf, alpha_f, TILE);
        // tmp_buf = -2*inner
        Muls(tmp_buf, tmp_buf, neg1_f, TILE);
        // tmp_buf = exp(...)
        Exp(tmp_buf, tmp_buf, TILE);
        // tmp_buf = 1 + exp(...)
        Adds(tmp_buf, tmp_buf, one_f, TILE);
        // tmp_buf = x / (1 + exp(...)) = GELU
        Div(tmp_buf, xf_buf, tmp_buf, TILE);

        // ---- 3. scalar copy float tmp → half yh → 写 GM ----
        for (int32_t j = 0; j < TILE; ++j) {
            yh_buf.SetValue(j, static_cast<half>(tmp_buf.GetValue(j)));
        }
        for (uint32_t j = 0; j < rem; ++j) {
            Yg.SetValue((uint64_t)base + j, yh_buf.GetValue((int32_t)j));
        }
    }
}
