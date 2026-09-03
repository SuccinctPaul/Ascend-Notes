// =============================================================================
// GELU 诊断 kernel: 逐步骤回写到 Y，确认每个 Vector 原语是否生效
//
// mode (从 tiling[1] 读取):
//   0 = IDENTITY:          y[i] = x[i]                    (仅 DataCopy + Cast)
//   1 = SQUARE:            y[i] = x[i]^2                  (Mul)
//   2 = MULS_BETA:         y[i] = 0.044715 * x[i]^3      (Mul + Muls)
//   3 = INNER_X_BETA:      y[i] = x + 0.044715*x^3       (Add)
//   4 = TWO_INNER:         y[i] = alpha*(x + beta*x^3)   (Muls)
//   5 = NEG_TWO_INNER:     y[i] = -alpha*(x + beta*x^3)  (Muls with neg1_f)
//   6 = EXP:               y[i] = exp(-2*inner)          (Exp)
//   7 = ONE_PLUS_EXP:      y[i] = 1 + exp(...)           (Adds)
//   8 = FINAL_DIV:         y[i] = x / (1+exp(...))       (Div)
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

static constexpr int32_t TILE = 256;

extern "C" __global__ __aicore__
void diag_gelu_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t N    = t[0];
    const uint32_t mode = t[1];

    GlobalTensor<half> Xg; GlobalTensor<half> Yg;
    Xg.SetGlobalBuffer((__gm__ half*)x, N);
    Yg.SetGlobalBuffer((__gm__ half*)y, N);

    LocalTensor<half>  xh_buf;    xh_buf.SetSize(TILE);
    LocalTensor<half>  yh_buf;    yh_buf.SetSize(TILE);
    LocalTensor<float> xf_buf;    xf_buf.SetSize(TILE);
    LocalTensor<float> tmp_buf;   tmp_buf.SetSize(TILE);
    LocalTensor<float> yf_buf;    yf_buf.SetSize(TILE);

    const float alpha_f = 1.5957691216057308f;
    const float beta_f  = 0.044715f;
    const float neg1_f  = -1.0f;
    const float one_f   = 1.0f;

    const uint32_t num_tiles = (N + TILE - 1u) / TILE;
    const uint32_t stride    = GetBlockNum();
    const uint32_t start     = GetBlockIdx();

    for (uint32_t bid = start; bid < num_tiles; bid += stride) {
        const uint32_t base = bid * TILE;
        const uint32_t rem  = (base + TILE <= N) ? TILE : (N - base);

        if (rem == TILE) {
            DataCopy(xh_buf, Xg[base], TILE);
        } else {
            for (int32_t j = 0; j < TILE; ++j) xh_buf.SetValue(j, half(0.0f));
            for (uint32_t j = 0; j < rem; ++j)
                xh_buf.SetValue((int32_t)j, Xg.GetValue((uint64_t)base + j));
        }

        for (int32_t j = 0; j < TILE; ++j)
            xf_buf.SetValue(j, static_cast<float>(xh_buf.GetValue(j)));

        LocalTensor<float>* out = &yf_buf;
        switch (mode) {
            case 0:   // IDENTITY
                for (int32_t j = 0; j < TILE; ++j)
                    yf_buf.SetValue(j, xf_buf.GetValue(j));
                break;
            case 1:   // SQUARE = x^2
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                out = &tmp_buf;
                break;
            case 2: { // 0.044715 * x^3
                Mul(tmp_buf, xf_buf, xf_buf, TILE);      // x^2
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);     // x^3
                Muls(tmp_buf, tmp_buf, beta_f, TILE);    // beta*x^3
                out = &tmp_buf;
                break;
            }
            case 3: { // x + beta*x^3
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                out = &tmp_buf;
                break;
            }
            case 4: { // alpha*(x + beta*x^3)  = 2*inner
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, alpha_f, TILE);
                out = &tmp_buf;
                break;
            }
            case 5: { // -2*inner
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, alpha_f, TILE);
                Muls(tmp_buf, tmp_buf, neg1_f, TILE);
                out = &tmp_buf;
                break;
            }
            case 6: { // exp(-2*inner)
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, alpha_f, TILE);
                Muls(tmp_buf, tmp_buf, neg1_f, TILE);
                Exp(tmp_buf, tmp_buf, TILE);
                out = &tmp_buf;
                break;
            }
            case 7: { // 1 + exp(-2*inner)
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, alpha_f, TILE);
                Muls(tmp_buf, tmp_buf, neg1_f, TILE);
                Exp(tmp_buf, tmp_buf, TILE);
                Adds(tmp_buf, tmp_buf, one_f, TILE);
                out = &tmp_buf;
                break;
            }
            case 8: { // x / (1 + exp(...)) = FINAL GELU
                Mul(tmp_buf, xf_buf, xf_buf, TILE);
                Mul(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, beta_f, TILE);
                Add(tmp_buf, xf_buf, tmp_buf, TILE);
                Muls(tmp_buf, tmp_buf, alpha_f, TILE);
                Muls(tmp_buf, tmp_buf, neg1_f, TILE);
                Exp(tmp_buf, tmp_buf, TILE);
                Adds(tmp_buf, tmp_buf, one_f, TILE);
                Div(yf_buf, xf_buf, tmp_buf, TILE);
                out = &yf_buf;
                break;
            }
            default:
                for (int32_t j = 0; j < TILE; ++j) yf_buf.SetValue(j, -999.0f);
                break;
        }

        // cast output → half → yh_buf
        for (int32_t j = 0; j < TILE; ++j)
            yh_buf.SetValue(j, static_cast<half>(out->GetValue(j)));

        if (rem == TILE) {
            DataCopy(Yg[base], yh_buf, TILE);
        } else {
            for (uint32_t j = 0; j < rem; ++j)
                Yg.SetValue((uint64_t)base + j, yh_buf.GetValue((int32_t)j));
        }
    }
}
