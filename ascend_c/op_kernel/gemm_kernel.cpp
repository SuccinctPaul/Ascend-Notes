#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__
void gemm_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                 GM_ADDR workspace, GM_ADDR tiling)
{
    uint32_t* t = reinterpret_cast<uint32_t*>(tiling);
    uint32_t M = t[0];
    uint32_t K = t[1];
    uint32_t N = t[2];

    GlobalTensor<float> A_global;
    GlobalTensor<float> B_global;
    GlobalTensor<float> C_global;

    A_global.SetGlobalBuffer((__gm__ float*)a, M*K);
    B_global.SetGlobalBuffer((__gm__ float*)b, K*N);
    C_global.SetGlobalBuffer((__gm__ float*)c, M*N);

    // 朴素 GEMM（先跑通）
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += A_global[m*K + k] * B_global[k*N + n];
            }
            C_global[m*N + n] = acc;
        }
    }
}

