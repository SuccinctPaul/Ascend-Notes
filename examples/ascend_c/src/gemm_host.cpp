// =============================================================================
// GEMM host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 职责:
//   1. 初始化 ACL (Ascend Computing Language) 运行时
//   2. 在 host 端生成 fp16 随机矩阵 A, B
//   3. H2D 拷贝到 device, 下发 tiling
//   4. 调用 aclrtlaunch_gemm_kernel() 启动 kernel
//   5. D2H 取回结果 C
//   6. 在 CPU 上算参考结果 (fp16 输入 + fp32 累加), 与 kernel 结果对比
//   7. 打印 PASS/FAIL 与最大误差
//
// ---- kernel 启动方式 ----
// 本版本使用官方 ascendc.cmake 框架 (ascendc_library STATIC)。
// 框架在编译时自动完成:
//   - bisheng 编译 device kernel → device.o
//   - 生成 host_stub.cpp (内含 device 二进制 + launch stub)
//   - ascendc_pack_kernel 打包 → lib/libgemm.a
//
// libgemm.a 导出 aclrtlaunch_gemm_kernel(), host 只需调用它即可启动 kernel。
// 该函数内部自动完成 (对 host 透明):
//   - RegisterAscendBinary : 把 device 二进制注册到 NPU runtime
//   - AllocAscendMemDevice : 分配 overflow 状态内存
//   - LaunchAscendKernel    : 下发 kernel 到 AI Core
//
// 类比 CUDA: aclrtlaunch_gemm_kernel ≈ CUDA 的 kernel<<<grid,block>>>(...) launch stub,
// 由 nvcc (此处是 bisheng + ascendc.cmake 框架) 自动生成。
//
// ---- 旧版手动 launch (已弃用, 保留注释供学习参考) ----
// 之前的版本手动调用低层 ACL API:
//   aclrtBinaryLoadFromFile → aclrtBinaryGetFunctionByEntry →
//   aclrtKernelArgsInit/Append/Finalize → aclrtLaunchKernelWithConfig
// 但 bisheng 直接编出的 raw .o 缺运行时 magic header, 需 ascendc_pack_kernel 打包。
// 现在用 ascendc.cmake 框架自动打包, host 代码大幅简化。
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"

// ---- host 侧 fp16 类型 ----
// aarch64 GCC 原生支持 __fp16; 用 half_t 别名, 与 kernel 端 half 对应
using half_t = __fp16;

// ---- ascendc.cmake 框架生成的 host 侧 launch 函数 (链接自 libgemm.a) ----
// 签名由 kernel 入口签名自动推导:
//   kernel: gemm_kernel(a, b, c, workspace, tiling)
//   stub  : aclrtlaunch_gemm_kernel(numBlocks, stream, a, b, c, workspace, tiling)
//
// 参数:
//   numBlocks : AI Core 并行核数 (本朴素版=1, 无多核并行)
//   stream    : ACL stream (异步执行队列)
//   a, b, c   : device 指针 (aclrtMalloc 分配的 GM 地址)
//   workspace : device workspace (本朴素版不用, 传 nullptr)
//   tiling    : device tiling buffer (含 M/K/N 三个 uint32)
//
// 返回值: 0=成功, 非 0=失败
extern "C" uint32_t aclrtlaunch_gemm_kernel(
    uint32_t numBlocks, void *stream,
    void *a, void *b, void *c,
    void *workspace, void *tiling);

// 简易错误检查宏: 失败时打印并退出
#define CHECK_ACL(expr, name)                                                  \
    do {                                                                       \
        aclError _e = (expr);                                                  \
        if (_e != ACL_SUCCESS) {                                               \
            std::cerr << "[FAIL] " << (name) << " error=" << _e                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// CPU 参考 GEMM: fp16 输入, fp32 累加, fp16 输出 (与 kernel 同精度策略)
static void gemm_cpu_ref(const half_t *A, const half_t *B, half_t *C,
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
    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, 0), "aclrtCreateContext");
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    // ---- 2. 矩阵规模与 host 数据 ----
    const int M = 128, K = 128, N = 128;
    std::vector<half_t> h_A(M * K), h_B(K * N), h_C(M * N);

    // 随机初始化 A, B: 取 [-1, 1] 区间, fp16 可表示范围内
    srand(0);
    auto randf = []() { return rand() / float(RAND_MAX) * 2.0f - 1.0f; };
    for (auto &x : h_A) x = half_t(randf());
    for (auto &x : h_B) x = half_t(randf());

    // ---- 3. device 内存分配 + H2D 拷贝 ----
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_ACL(aclrtMalloc(&d_A, M * K * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc A");
    CHECK_ACL(aclrtMalloc(&d_B, K * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc B");
    CHECK_ACL(aclrtMalloc(&d_C, M * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc C");

    CHECK_ACL(aclrtMemcpy(d_A, M * K * sizeof(half_t), h_A.data(), M * K * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE), "H2D A");
    CHECK_ACL(aclrtMemcpy(d_B, K * N * sizeof(half_t), h_B.data(), K * N * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE), "H2D B");

    // ---- 4. 下发 tiling (M, K, N) ----
    // tiling 是一段 device 可见内存, kernel 通过第 5 个参数读取
    uint32_t tiling[3] = {uint32_t(M), uint32_t(K), uint32_t(N)};
    void *d_tiling = nullptr;
    CHECK_ACL(aclrtMalloc(&d_tiling, sizeof(tiling), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc tiling");
    CHECK_ACL(aclrtMemcpy(d_tiling, sizeof(tiling), tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE), "H2D tiling");

    // ---- 5. 启动 kernel ----
    // aclrtlaunch_gemm_kernel 内部自动:
    //   RegisterAscendBinary → 构造 args (含 overflow 状态内存) → LaunchAscendKernel
    // blockDim=1: 朴素版只用一个核 (无多核并行)
    uint32_t ret = aclrtlaunch_gemm_kernel(
        /*numBlocks=*/1, stream,
        d_A, d_B, d_C,
        /*workspace=*/nullptr, d_tiling);
    if (ret != 0) {
        std::cerr << "[FAIL] aclrtlaunch_gemm_kernel ret=" << ret << std::endl;
        std::exit(1);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    // ---- 6. D2H 取回结果 ----
    CHECK_ACL(aclrtMemcpy(h_C.data(), M * N * sizeof(half_t), d_C, M * N * sizeof(half_t), ACL_MEMCPY_DEVICE_TO_HOST),
              "D2H C");

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
    CHECK_ACL(aclrtFree(d_A), "aclrtFree A");
    CHECK_ACL(aclrtFree(d_B), "aclrtFree B");
    CHECK_ACL(aclrtFree(d_C), "aclrtFree C");
    CHECK_ACL(aclrtFree(d_tiling), "aclrtFree tiling");
    CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream");
    CHECK_ACL(aclrtDestroyContext(ctx), "aclrtDestroyContext");
    CHECK_ACL(aclrtResetDevice(0), "aclrtResetDevice");
    CHECK_ACL(aclFinalize(), "aclFinalize");

    return pass ? 0 : 1;
}
