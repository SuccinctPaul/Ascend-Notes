// =============================================================================
// GEMM host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 职责:
//   1. 初始化 ACL (Ascend Computing Language) 运行时
//   2. 在 host 端生成 fp16 随机矩阵 A, B
//   3. H2D 拷贝到 device, 下发 tiling
//   4. 加载 kernel 二进制 (gemm_kernel.o), 取得 funcHandle, 打包 args, 启动 kernel
//   5. D2H 取回结果 C
//   6. 在 CPU 上算参考结果 (fp16 输入 + fp32 累加), 与 kernel 结果对比
//   7. 打印 PASS/FAIL 与最大误差
//
// ACL kernel launch 关键概念 (CANN 9.0.0 现代写法):
//   - aclrtBinaryLoadFromFile : 把 bisheng 编出的 .o 载入 device runtime
//   - aclrtBinaryGetFunctionByEntry : 按 tilingKey 取得函数句柄 (本朴素版 tilingKey=0)
//   - aclrtKernelArgsInit/Append/Finalize : 显式构造 args (支持 host-mem 入参等高级特性)
//   - aclrtLaunchKernelWithConfig : 用 funcHandle + argsHandle 启动 kernel
//   - 910B cube kernel 额外要求: args 头部追加一个 FFTS 控制地址 (rtGetC2cCtrlAddr)
//
// 类比 CUDA: aclrtBinaryLoadFromFile ≈ cuModuleLoad, aclrtBinaryGetFunctionByEntry ≈
// cuModuleGetFunction, aclrtLaunchKernelWithConfig ≈ cuLaunchKernel.
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include "acl/acl.h"
#include "acl/acl_rt.h"   // aclrtBinaryLoadFromFile / aclrtLaunchKernelWithConfig 等

// ---- host 侧 fp16 类型 ----
// aarch64 GCC 原生支持 __fp16; 用 half_t 别名, 与 kernel 端 half 对应
using half_t = __fp16;

// ---- ACL 头文件未导出、但运行时库提供的两个底层符号 ----
// rtGetC2cCtrlAddr : 取 FFTS (Fast Task Schedule) 控制地址, 910B cube kernel 启动必需
// aclrtGetSocName  : 查询当前 SoC 名字 (如 "Ascend910B2"), 用于判断是否需要 FFTS prepend
extern "C" {
typedef uint32_t rtError_t;
rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *fftsLen);
const char *aclrtGetSocName();
}

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

int main(int argc, char** argv) {
    // kernel 二进制路径: 默认 cwd 下的 gemm_kernel.o, 也可用 argv[1] 指定
    std::string kernel_path = (argc > 1) ? argv[1] : "gemm_kernel.o";

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
    for (auto& x : h_A) x = half_t(randf());
    for (auto& x : h_B) x = half_t(randf());

    // ---- 3. device 内存分配 + H2D 拷贝 ----
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_ACL(aclrtMalloc(&d_A, M * K * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc A");
    CHECK_ACL(aclrtMalloc(&d_B, K * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc B");
    CHECK_ACL(aclrtMalloc(&d_C, M * N * sizeof(half_t), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc C");

    CHECK_ACL(aclrtMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE), "H2D A");
    CHECK_ACL(aclrtMemcpy(d_B, h_B.data(), K * N * sizeof(half_t), ACL_MEMCPY_HOST_TO_DEVICE), "H2D B");

    // ---- 4. 下发 tiling (M, K, N) ----
    // tiling 是一段 device 可见内存, kernel 通过第 5 个参数读取
    uint32_t tiling[3] = {uint32_t(M), uint32_t(K), uint32_t(N)};
    void* d_tiling = nullptr;
    CHECK_ACL(aclrtMalloc(&d_tiling, sizeof(tiling), ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc tiling");
    CHECK_ACL(aclrtMemcpy(d_tiling, tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE), "H2D tiling");

    // ---- 5. 加载 kernel 二进制并取得 funcHandle ----
    // aclrtBinaryLoadFromFile: 把 .o 注册到 device runtime, 返回 binHandle
    // aclrtBinaryGetFunctionByEntry: 按 tilingKey 取函数句柄
    //   - tilingKey 是 host 下发的"tiling 方案编号", 用于让同一 kernel 源码支持多种分块
    //   - 本朴素 kernel 不分块, tilingKey=0 (对应 bisheng_intf.cmake 里的 TILING_KEY_VAR=0)
    aclrtBinHandle binHandle = nullptr;
    CHECK_ACL(aclrtBinaryLoadFromFile(kernel_path.c_str(), nullptr, &binHandle), "aclrtBinaryLoadFromFile");

    aclrtFuncHandle funcHandle = nullptr;
    const uint64_t tilingKey = 0;
    CHECK_ACL(aclrtBinaryGetFunctionByEntry(binHandle, tilingKey, &funcHandle),
              "aclrtBinaryGetFunctionByEntry");

    // ---- 6. 构造 args (aclrtKernelArgsInit/Append/Finalize) ----
    // 这是 CANN 9.0 推荐的 args 打包方式, 支持后续 host-mem 入参等高级特性。
    // args 内容: 一组 void* (每个 8 字节), 顺序与 kernel 签名 (a, b, c, workspace, tiling) 一致。
    aclrtArgsHandle argsHandle = nullptr;
    CHECK_ACL(aclrtKernelArgsInit(funcHandle, &argsHandle), "aclrtKernelArgsInit");

    // 910B cube (非 AIV) kernel 额外要求: args 头部追加 FFTS 控制地址。
    // 这是 910B 硬件 Fast Task Schedule 机制 —— cube kernel 启动时需要一段 FFTS 控制内存。
    // 官方 mskl launcher 在 LaunchKernel 前会调 rtGetC2cCtrlAddr 并把地址作为 args[0] 插入。
    // 我们照做: 查 SoC 名, 若是 910B 且 kernel 不是 AIV (本朴素 kernel 是 AIC/cube), 则 prepend。
    const char* soc = aclrtGetSocName();
    bool is_910b = (soc && std::strstr(soc, "Ascend910B") != nullptr);
    bool is_aiv = false;  // 本朴素 kernel 编译用 dav-c220-cube (AIC, 含 cube+vec), 不是纯 AIV

    std::vector<void*> args;
    args.reserve(6);
    if (is_910b && !is_aiv) {
        uint64_t ffts_addr = 0;
        uint32_t ffts_len = 0;
        rtError_t r = rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
        if (r != 0 || ffts_addr == 0) {
            std::cerr << "[FAIL] rtGetC2cCtrlAddr ret=" << r
                      << " addr=0 (soc=" << (soc ? soc : "null") << ")" << std::endl;
            std::exit(1);
        }
        args.push_back(reinterpret_cast<void*>(ffts_addr));
    }

    // kernel 签名: gemm_kernel(a, b, c, workspace, tiling)
    // workspace: 本朴素版不用, 传 nullptr
    args.push_back(d_A);
    args.push_back(d_B);
    args.push_back(d_C);
    args.push_back(nullptr);     // workspace
    args.push_back(d_tiling);

    // aclrtKernelArgsAppend: 把 args buffer (一组 void*) 一次性追加到 argsHandle
    // paramSize = args.size() * sizeof(void*) (每个 void* 8 字节)
    aclrtParamHandle paramHandle = nullptr;
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, args.data(),
                                    args.size() * sizeof(void*), &paramHandle),
              "aclrtKernelArgsAppend");
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle), "aclrtKernelArgsFinalize");

    // ---- 7. 启动 kernel ----
    // blockDim=1: 朴素版只用一个核 (无多核并行)
    // cfg=nullptr: 不使用额外 launch 配置 (如 L2 preload 等)
    CHECK_ACL(aclrtLaunchKernelWithConfig(funcHandle, /*blockDim=*/1, stream,
                                          /*cfg=*/nullptr, argsHandle, /*reserve=*/nullptr),
              "aclrtLaunchKernelWithConfig");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    // ---- 8. D2H 取回结果 ----
    CHECK_ACL(aclrtMemcpy(h_C.data(), d_C, M * N * sizeof(half_t), ACL_MEMCPY_DEVICE_TO_HOST),
              "D2H C");

    // ---- 9. CPU 参考计算 + 误差校验 ----
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
              << ", M=N=K=" << M << ", dtype=fp16"
              << ", soc=" << (soc ? soc : "unknown")
              << ", ffts_prepended=" << (is_910b && !is_aiv ? "yes" : "no")
              << ", args_n=" << args.size() << ")" << std::endl;

    // ---- 10. 资源释放 ----
    // 注: argsHandle 无显式 destroy API, 进程退出即回收
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
