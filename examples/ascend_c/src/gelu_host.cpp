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

// ---- ascendc.cmake 生成的 launch 函数 (链接自 libgelu.a) ----
//   kernel: gelu_kernel(x, y, workspace, tiling)
//   stub  : aclrtlaunch_gelu_kernel(numBlocks, stream, x, y, workspace, tiling)
extern "C" int aclrtlaunch_gelu_kernel(uint32_t numBlocks, aclrtStream stream,
                                        void* x, void* y, void* workspace, void* tiling);

// ---- GELU tanh 近似常数 ----
static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
static constexpr double CUBIC_COEF     = 0.044715;

static float gelu_host_ref(float xv) {
    double x = xv;
    const double inner = SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x);
    return static_cast<float>(x * 0.5 * (1.0 + std::tanh(inner)));
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

    // ---- 4. device buffer ----
    const size_t nbytes = size_t(N) * sizeof(half_t);
    void* d_x = nullptr; check("aclrtMalloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("aclrtMalloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t tiling_val = N;
    void* d_tile = nullptr;
    check("aclrtMalloc tiling", aclrtMalloc(&d_tile, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));

    // H2D
    check("aclrtMemcpy H2D x",
          aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D tiling",
          aclrtMemcpy(d_tile, sizeof(uint32_t), &tiling_val, sizeof(uint32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 5. 下发 kernel ----
    // kernel TILE=256 half (Vector Gelu<float> + scalar h/f cast), grid-stride.
    static constexpr uint32_t KERNEL_TILE = 256u;
    const uint32_t numBlocks = std::min<uint32_t>(
        32768u,
        (N + KERNEL_TILE - 1u) / KERNEL_TILE
    );
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_gelu_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    if (rc != 0) { std::cerr << "aclrtlaunch_gelu_kernel returned " << rc << "\n"; return 2; }
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
              << "kernel ms    = " << ms << " (含同步，仅粗测)" << std::endl
              << "max_abs_err  = " << max_abs << std::endl
              << "bad_elements = " << bad << " / " << N << std::endl
              << "result       = " << (pass ? "PASS" : "FAIL") << std::endl;

    // ---- 8. 清理 ----
    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
