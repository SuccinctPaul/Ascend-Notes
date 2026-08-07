// =============================================================================
// GEMM host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 职责:
//   1. 初始化 ACL (Ascend Computing Language) 运行时
//   2. 在 host 端生成 fp16 随机矩阵 A, B
//   3. H2D 拷贝到 device, 下发 tiling, 启动 gemm_kernel
//   4. D2H 取回结果 C
//   5. 在 CPU 上算参考结果 (fp16 输入 + fp32 累加), 与 kernel 结果对比
//   6. 打印 PASS/FAIL 与最大误差
//
// ACL 关键概念:
//   - aclrtSetDevice / aclrtCreateContext / aclrtCreateStream:
//     设备 / 上下文 / 流, NPU 程序的标准三件套
//   - aclrtMalloc: 在 device HBM 上分配
//   - aclrtMemcpy: H2D / D2H 数据搬运 (类比 cudaMemcpy)
//   - aclrtLaunchKernel: 启动 device kernel (类比 cudaLaunchKernel)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include "acl/acl.h"

// ---- host 侧 fp16 类型 ----
// aarch64 GCC 原生支持 __fp16; 用 half_t 别名, 与 kernel 端 half 对应
using half_t = __fp16;

// CPU 参考 GEMM: fp16 输入, fp32 累加, fp16 输出 (与 kernel 同精度策略)
static void gemm_cpu_ref(const half_t* A, const half_t* B, half_t* C,
                         int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += float(A[i * K + k]) * float(B[k * N + j]);
            }
            C[i * N + j] = half_t(acc);
        }
    }
}

int main() {
    // ---- 1. ACL 运行时初始化 ----
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx;
    aclrtCreateContext(&ctx, 0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // ---- 2. 矩阵规模与 host 数据 ----
    const int M = 128, K = 128, N = 128;
    std::vector<half_t> h_A(M * K), h_B(K * N), h_C(M * N);

    // 随机初始化 A, B: 取 [-1, 1] 区间, fp16 可表示范围内
    srand(0);
    auto randf = []() { return rand() / float(RAND_MAX) * 2.0f - 1.0f; };
    for (auto& x : h_A) x = half_t(randf());
    for (auto& x : h_B) x = half_t(randf());

    // ---- 3. device 内存分配 + H2D 拷贝 ----
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    aclrtMalloc(&d_A, M * K * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&d_B, K * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&d_C, M * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY);

    aclrtMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_B, h_B.data(), K * N * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // ---- 4. 下发 tiling (M, K, N) ----
    // tiling 是一段 device 可见内存, kernel 启动时通过第 5 个参数读取
    uint32_t tiling[3] = {uint32_t(M), uint32_t(K), uint32_t(N)};
    void* d_tiling = nullptr;
    aclrtMalloc(&d_tiling, sizeof(tiling), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(d_tiling, tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // ---- 5. 启动 kernel ----
    // args 顺序必须与 kernel 签名 (a, b, c, workspace, tiling) 一致
    // block_x=1, block_y=1: 朴素版只用一个核 (无多核并行)
    void* args[] = {d_A, d_B, d_C, nullptr, d_tiling};
    aclrtLaunchKernel("gemm_kernel", 1, 1, args, 0, stream);
    aclrtSynchronizeStream(stream);

    // ---- 6. D2H 取回结果 ----
    aclrtMemcpy(h_C.data(), d_C, M * N * sizeof(half_t), ACL_MEMCPY_DEVICE_TO_HOST);

    // ---- 7. CPU 参考计算 + 误差校验 ----
    std::vector<half_t> h_Cref(M * N);
    gemm_cpu_ref(h_A.data(), h_B.data(), h_Cref.data(), M, K, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(float(h_C[i]) - float(h_Cref[i]));
        if (diff > max_err) max_err = diff;
    }
    // fp16 容差 atol=1e-2 (kernel 与 CPU 参考用同精度策略, 误差应极小)
    bool pass = (max_err < 1e-2f);

    std::cout << "ascend_c GEMM: " << (pass ? "PASS" : "FAIL")
              << " (max_abs_error=" << max_err
              << ", M=N=K=" << M << ", dtype=fp16)" << std::endl;

    // ---- 8. 资源释放 ----
    aclrtFree(d_A);
    aclrtFree(d_B);
    aclrtFree(d_C);
    aclrtFree(d_tiling);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();

    return pass ? 0 : 1;
}
