// GELU v3 host: 使用新 target 名 aclrtlaunch_gelu_v3_kernel, 避免旧 ExternalProject 二进制缓存
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include "acl/acl.h"
using half_t = __fp16;

extern "C" int aclrtlaunch_gelu_v3_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) { std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << "\n"; std::exit(1); }
}
static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
static constexpr double CUBIC_COEF     = 0.044715;
static float gelu_host_ref(float xv) {
    double x = xv;
    const double inner = SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x);
    const double neg_twice = -2.0 * inner;
    const double den = 1.0 + std::exp(neg_twice);
    return static_cast<float>(x / den);
}

int main(int argc, char** argv) {
    const uint32_t N = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8192u;
    check("aclInit", aclInit(nullptr));
    check("aclrtSetDevice", aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    std::mt19937 rng(20250417u);
    std::uniform_real_distribution<float> dist(-4.0f, 5.0f);
    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N), y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
        y_ref[i] = static_cast<half_t>(gelu_host_ref(static_cast<float>(x_h[i])));
    }
    const size_t nbytes = size_t(N) * sizeof(half_t);
    void *d_x = nullptr, *d_y = nullptr;
    check("Malloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("Malloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t tiling[1] = {N};
    void* d_tile = nullptr;
    check("Malloc tiling", aclrtMalloc(&d_tile, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2D tiling", aclrtMemcpy(d_tile, sizeof(tiling), tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    check("H2D x", aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t numBlocks = 1u;  // CANN 9 云容器多 block 调度不可靠, 强制 1 block 跑全量
    const auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_gelu_v3_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    if (rc != 0) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<half_t> y_dev(N);
    check("D2H y", aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

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
    std::cout << "=== Ascend C GELU (v3, EXP formula, softmax-fsub pattern) ===" << std::endl
              << "N            = " << N << std::endl
              << "kernel ms    = " << ms << " (含同步，仅粗测)" << std::endl
              << "max_abs_err  = " << max_abs << std::endl
              << "bad_elements = " << bad << " / " << N << std::endl
              << "result       = " << ((bad == 0) ? "PASS" : "FAIL") << std::endl;

    if (N <= 32u) {
        std::cout << "--- DEBUG N <= 32: i / x / ref / dev ---" << std::endl;
        for (uint32_t i = 0; i < N; ++i) {
            std::cout << "  i=" << i
                      << "  x="   << static_cast<float>(x_h[i])
                      << "  ref=" << static_cast<float>(y_ref[i])
                      << "  dev=" << static_cast<float>(y_dev[i]) << std::endl;
        }
    }

    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
