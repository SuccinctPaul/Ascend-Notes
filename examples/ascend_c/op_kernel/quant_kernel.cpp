// =============================================================================
// INT8 对称量化 kernels — Ascend C (CANN 原生)
//
// 对应 docs/ops/08-quantization.md §2.1 (per-row scale, §5.3):
//   quant:   scale[r] = max( absmax(x[r,:])/127, eps )        (fp32 归约)
//            q[r,c]   = cast_round( x[r,c]/scale[r] ) → int8
//   dequant: y[r,c]   = q[r,c] * scale[r]   (§5.4 反量化在 Vector/UB 做)
//
// 实现规范 (与 rmsnorm_kernel 一致):
//   - tiling 常数经 GlobalTensor<float>.GetValue 标量读;
//   - Cast 内建指令的工作张量用 TPipe/TBuf 分配真实 UB (裸 LocalTensor 无后备);
//   - aicore 禁止整型变量↔浮点变量 cast, int8↔float 转换全部走 Cast 内建;
//   - Host 下发 numBlocks=1 (CANN 9 多 block 调度规避, 同 softmax)。
// =============================================================================
#include "kernel_operator.h"
using namespace AscendC;

// ---------------------------------------------------------------------------
// quant: x (rows×D, fp16) → q (rows×D, int8) + scale (rows, fp32)
// tiling cf: cf[0]=127.0, cf[1]=1e-12 (防除零下限)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__
void quant_kernel(GM_ADDR x, GM_ADDR q, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];
    const uint32_t D        = T[1];
    GlobalTensor<float> Cf;
    Cf.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(T + 4), 8u);
    const float CQMAX = Cf.GetValue(0);   // 127.0
    const float CEPS  = Cf.GetValue(1);   // 1e-12

    GlobalTensor<half> Xg;
    Xg.SetGlobalBuffer((__gm__ half*)x, uint64_t(num_rows) * D);
    GlobalTensor<int8_t> Qg;
    Qg.SetGlobalBuffer((__gm__ int8_t*)q, uint64_t(num_rows) * D);
    GlobalTensor<float> SG;
    SG.SetGlobalBuffer((__gm__ float*)scale, uint64_t(num_rows));

    // Cast 工作张量: 真实 UB (fp32 槽 + int8 槽)
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufV, bufH, bufQ;
    pipe.InitBuffer(bufV, 32);
    pipe.InitBuffer(bufH, 32);
    pipe.InitBuffer(bufQ, 32);
    LocalTensor<float> sVAL = bufV.Get<float>(8);
    LocalTensor<half>  sH16 = bufH.Get<half>(8);
    LocalTensor<int8_t> sQ8 = bufQ.Get<int8_t>(8);

    for (uint64_t row = 0ull; row < (uint64_t)num_rows; ++row) {
        const uint64_t base = row * (uint64_t)D;
        // ---- Pass 1: fp32 归约 absmax ----
        float amax = 0.0f;
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            const float av = static_cast<float>(Xg.GetValue(base + c));
            const float a  = av < 0.0f ? -av : av;
            if (a > amax) amax = a;
        }
        // ---- scale = max(absmax/127, eps), fp32 下发 ----
        float scale_f = amax / CQMAX;
        if (scale_f < CEPS) scale_f = CEPS;
        SG.SetValue(row, scale_f);
        // ---- Pass 2: q = cast_round(x / scale) → int8 (无 clamp: x/scale ∈ [-127,127]) ----
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sVAL.SetValue(0, static_cast<float>(Xg.GetValue(base + c)) / scale_f);
            // dav-c220 无 fp32↔int8 直转: fp32 → fp16 → int8 (CAST_RINT = 四舍六入五成双)
            Cast(sH16, sVAL, RoundMode::CAST_NONE, 8);
            Cast(sQ8, sH16, RoundMode::CAST_RINT, 8);
            Qg.SetValue(base + c, sQ8.GetValue(0));
        }
    }
}

// ---------------------------------------------------------------------------
// dequant: q (rows×D, int8) × scale (rows, fp32) → y (rows×D, fp16)
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__
void dequant_kernel(GM_ADDR q, GM_ADDR scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    __gm__ uint32_t* T = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const uint32_t num_rows = T[0];
    const uint32_t D        = T[1];

    GlobalTensor<int8_t> Qg;
    Qg.SetGlobalBuffer((__gm__ int8_t*)q, uint64_t(num_rows) * D);
    GlobalTensor<float> SG;
    SG.SetGlobalBuffer((__gm__ float*)scale, uint64_t(num_rows));
    GlobalTensor<half> Yg;
    Yg.SetGlobalBuffer((__gm__ half*)y, uint64_t(num_rows) * D);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufQ, bufH, bufF;
    pipe.InitBuffer(bufQ, 32);
    pipe.InitBuffer(bufH, 32);
    pipe.InitBuffer(bufF, 32);
    LocalTensor<int8_t> sQ8 = bufQ.Get<int8_t>(8);
    LocalTensor<half>  sH16 = bufH.Get<half>(8);
    LocalTensor<float> sF   = bufF.Get<float>(8);

    for (uint64_t row = 0ull; row < (uint64_t)num_rows; ++row) {
        const uint64_t base = row * (uint64_t)D;
        const float scale_f = SG.GetValue(row);
        for (uint64_t c = 0ull; c < (uint64_t)D; ++c) {
            sQ8.SetValue(0, Qg.GetValue(base + c));
            // dav-c220 无 fp32↔int8 直转: int8 → fp16 → fp32
            Cast(sH16, sQ8, RoundMode::CAST_NONE, 8);
            Cast(sF, sH16, RoundMode::CAST_NONE, 8);
            const float yf = sF.GetValue(0) * scale_f;
            Yg.SetValue(base + c, static_cast<half>(yf));
        }
    }
}
