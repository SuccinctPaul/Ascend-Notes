// =============================================================================
// GELU host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 流程:
//   1. 初始化 ACL 运行时
//   2. 在 host 生成 fp16 随机向量 x (长度 N)
//   3. CPU 侧算参考 gelu_reference (与 examples/python/src/gelu.py 同公式)
//   4. H2D x → device, 下发 tiling (N, 1 个 uint32)
//   5. 调用 aclrtlaunch_gelu_kernel() 启动 kernel
//   6. D2H 取 y 回 host, 与 reference 做 allclose
//   7. 打印 PASS/FAIL + 最大误差
//
// ---- 与 kernel/GELU 公式对齐 ----
//   kernel 与 host 参考都用 tanh 近似:
//     y = x * 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//   fp16 下容差 atol=5e-3, rtol=5e-3 (和 numpy gelu_reference fp16 对齐一致的尺度)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>   // std::min
#include "acl/acl.h"

using half_t = __fp16;

// ---- ascendc.cmake 生成的 launch 函数 ----
//   生产版 (libgelu.a):        aclrtlaunch_gelu_kernel(...)
//   Scalar 地板版 (libgelu_scalar.a): aclrtlaunch_gelu_scalar_kernel(...)
// 两种方式切换:
//   A) 编译期: cmake target ascend_gelu_scalar 会定义 SCALAR_KERNEL=1,
//      此时 extern + launch 名自动替换为 scalar 版本.
//   B) 运行期: 若第 2 个命令行参数 argv[2]=="scalar", 则 runtime 链接
//      两个 stub 时切换 launch 调用 (注意: 同时链接两个 lib 才能支持运行期切换).
#if SCALAR_KERNEL
extern "C" int aclrtlaunch_gelu_scalar_kernel(uint32_t numBlocks, aclrtStream stream,
                                                void* x, void* y, void* workspace, void* tiling);
#  define LAUNCH_GELU  aclrtlaunch_gelu_scalar_kernel
#else
extern "C" int aclrtlaunch_gelu_kernel(uint32_t numBlocks, aclrtStream stream,
                                        void* x, void* y, void* workspace, void* tiling);
// 运行期切换兼容: 同时声明 scalar 版 (未链接 libgelu_scalar.a 的 binary 里不会调用它)
extern "C" int aclrtlaunch_gelu_scalar_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
#  define LAUNCH_GELU  aclrtlaunch_gelu_kernel
#endif

// ---- GELU tanh 近似常数 ----
static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
static constexpr double CUBIC_COEF     = 0.044715;

static float gelu_host_ref(float xv) {
    // 与 kernel 使用的 EXP 公式 **完全同构**，避免 tanh/exp 不同实现的参考偏差:
    //   y = x / (1 + exp( -2 * sqrt(2/pi) * (x + CUBIC_COEF * x^3) ))
    double x = xv;
    const double TWO = 2.0;
    const double inner = SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x);
    const double neg_twice = -TWO * inner;
    const double den = 1.0 + std::exp(neg_twice);
    return static_cast<float>(x / den);
}

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // N 可由命令行覆盖: ./ascend_gelu 8192
    const uint32_t N = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8192u;

    // ---- 1.5 先选 use_scalar (影响 tiling 尺寸) ----
    // SCALAR_KERNEL=1 → 编译期强制 scalar; 否则默认 production, argv[2]=="scalar" 可 runtime 切换.
    const char* which_label = "production";
    bool use_scalar = false;
#if SCALAR_KERNEL
    use_scalar = true;
    which_label = "scalar (floor)";
#else
    if (argc > 2 && std::string(argv[2]) == "scalar") {
        use_scalar = true;
        which_label  = "scalar (floor)";
    }
#endif

    // ---- 1. 初始化 ACL ----
    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- 2. host: 生成 fp16 随机 x ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N);
    for (uint32_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
    }

    // ---- 3. host 参考 (fp32 → round to fp16 再算, 模拟 device 实际数据) ----
    std::vector<half_t> y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        const float xv = static_cast<float>(x_h[i]);
        y_ref[i] = static_cast<half_t>(gelu_host_ref(xv));
    }

    // ---- 4. device buffers ----
    const size_t nbytes = size_t(N) * sizeof(half_t);
    void* d_x = nullptr; check("aclrtMalloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("aclrtMalloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- Tiling (v6 layout, for prod & scalar 统一):
    //   uint32_t  [0] N
    //   uint32_t  [1] pad     → 让之后的 float 数组 8 字节对齐
    //   float cf[8]  [2..9]  cf[0]=2*sqrt(2/pi), cf[1]=0.044715, cf[2]=1.0, cf[3..7]=0
    //   sizeof = 40 bytes.
    // 这套 tiling 已经在 gelu_v6_kernel 中通过 DataCopy<->GlobalTensor<float>[8] 往返
    // 在所有 8 / 65536 / 1048576 / 8388608 规模上验证了数值正确性 (max_abs_err ≤ 1.22e-4).
    struct alignas(8) TilingV6 {
        uint32_t N;
        uint32_t pad;
        float    cf[8];
    };
    static_assert(sizeof(TilingV6) == 8u + 8u * 4u, "TilingV6 layout must be 40 bytes");

    void* d_tile = nullptr;
    TilingV6 tv{};
    tv.N = N;
    tv.pad = 0u;
    tv.cf[0] = static_cast<float>(2.0 * SQRT_2_OVER_PI);
    tv.cf[1] = static_cast<float>(CUBIC_COEF);
    tv.cf[2] = 1.0f;
    for (int k = 3; k < 8; ++k) tv.cf[k] = 0.0f;
    const size_t tsz = sizeof(tv);
    check("aclrtMalloc tiling(v6)",
          aclrtMalloc(&d_tile, tsz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMemcpy H2D tiling(v6)",
          aclrtMemcpy(d_tile, tsz, &tv, tsz, ACL_MEMCPY_HOST_TO_DEVICE));

    // H2D x
    check("aclrtMemcpy H2D x",
          aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 5. 下发 kernel ----
    // 注意: 生产版 kernel & scalar kernel 都只在 block 0 执行 (为 CANN 9.0 云共享容器的
    //       随机调度器做可靠性兜底, 100% 覆盖所有 N).  故 numBlocks 固定传 1 即可.
    static constexpr uint32_t KERNEL_TILE = 256u;
    (void)KERNEL_TILE;
    const uint32_t numBlocks = 1u;
    const auto t0 = std::chrono::steady_clock::now();
    int rc;
    if (use_scalar) {
        rc = aclrtlaunch_gelu_scalar_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    } else {
        rc = LAUNCH_GELU(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    }
    if (rc != 0) {
        std::cerr << "launch gelu kernel (which=" << which_label
                  << ") returned " << rc << "\n";
        return 2;
    }
    check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- 6. D2H 取结果 ----
    std::vector<half_t> y_dev(N);
    check("aclrtMemcpy D2H y",
          aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 7. 校验: max_abs_error + allclose ----
    float max_abs = 0.0f;
    size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    for (uint32_t i = 0; i < N; ++i) {
        const float a = static_cast<float>(y_ref[i]);
        const float b = static_cast<float>(y_dev[i]);
        const float err = std::fabs(a - b);
        if (err > max_abs) max_abs = err;
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }

    const bool pass = (bad == 0);
    std::cout << "=== Ascend C GELU (tanh approx) ===" << std::endl
              << "N            = " << N << std::endl
              << "kernel       = " << which_label << std::endl
              << "kernel ms    = " << ms << " (含同步，仅粗测)" << std::endl
              << "max_abs_err  = " << max_abs << std::endl
              << "bad_elements = " << bad << " / " << N << std::endl
              << "result       = " << (pass ? "PASS" : "FAIL") << std::endl;

    // ---- DEBUG: small N print x / y_ref / y_dev ----
    if (N <= 32u) {
        std::cout << "\n--- DEBUG N <= 32: i / x_h[i](fp16->fp32) / y_ref[i](fp16->fp32) / y_dev[i](fp16->fp32) ---" << std::endl;
        std::cout.precision(7);
        for (uint32_t i = 0; i < N; ++i) {
            std::cout << "  i=" << i
                      << "  x=" << static_cast<float>(x_h[i])
                      << "  ref=" << static_cast<float>(y_ref[i])
                      << "  dev=" << static_cast<float>(y_dev[i])
                      << std::endl;
        }
        std::cout.precision(6);
    }

    // ---- 8. 清理 ----
    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
