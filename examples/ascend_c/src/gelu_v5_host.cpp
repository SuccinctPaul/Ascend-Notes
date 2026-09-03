// gelu_v5_host: 构造 tiling [N, pad, CBIG, CCUB, CONE] 单位字节; 调用 gelu_v5_kernel.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include "acl/acl.h"
using half_t = __fp16;

extern "C" int aclrtlaunch_gelu_v5_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
#if USE_KERNEL_V6
extern "C" int aclrtlaunch_gelu_v6_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
#  define LAUNCH_FN aclrtlaunch_gelu_v6_kernel
#else
#  define LAUNCH_FN aclrtlaunch_gelu_v5_kernel
#endif

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) { std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << "\n"; std::exit(1); }
}
static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
static constexpr double CUBIC_COEF     = 0.044715;
static float gelu_host_ref(float xv) {
    double x = xv;
    const double neg_twice = -2.0 * (SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x));
    return static_cast<float>(x / (1.0 + std::exp(neg_twice)));
}

// Tiling layout: [N(u32), pad(u32), CBIG(f32), CCUB(f32), CONE(f32), 5 * unused f32 (padding for 8 floats DataCopy))]
// sizeof = 8 + 8*4 = 40 bytes.
struct alignas(8) TilingV5 {
    uint32_t N;
    uint32_t pad;
    float    cf[8];
};

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
    TilingV5 tile{};
    tile.N = N; tile.pad = 0u;
    tile.cf[0] = (float)(2.0 * SQRT_2_OVER_PI);
    tile.cf[1] = (float)CUBIC_COEF;
    tile.cf[2] = 1.0f;
    for (int k = 3; k < 8; ++k) tile.cf[k] = 0.0f;

    const size_t nbytes = size_t(N) * sizeof(half_t);
    void *d_x = nullptr, *d_y = nullptr, *d_t = nullptr;
    check("Mx", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("My", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("Mt", aclrtMalloc(&d_t, sizeof(tile), ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2Dt", aclrtMemcpy(d_t, sizeof(tile), &tile, sizeof(tile), ACL_MEMCPY_HOST_TO_DEVICE));
    check("H2Dx", aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t numBlocks = 1u;
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = LAUNCH_FN(numBlocks, stream, d_x, d_y, nullptr, d_t);
    if (rc != 0) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<half_t> y_dev(N);
    check("D2Hy", aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    float max_abs = 0.0f; size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    for (uint32_t i = 0; i < N; ++i) {
        const float a = static_cast<float>(y_ref[i]);
        const float b = static_cast<float>(y_dev[i]);
        const float err = std::fabs(a - b);
        if (err > max_abs) max_abs = err;
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }
    std::cout << "=== Ascend C GELU (v5, constants from tiling via DataCopy) ===" << std::endl
              << "N            = " << N << std::endl
              << "kernel ms    = " << ms << " (含同步，仅粗测)" << std::endl
              << "max_abs_err  = " << max_abs << std::endl
              << "bad_elements = " << bad << " / " << N << std::endl
              << "result       = " << ((bad == 0) ? "PASS" : "FAIL") << std::endl;
    if (N <= 32u) {
        std::cout << "--- DEBUG: i / x / ref / dev ---" << std::endl;
        for (uint32_t i = 0; i < N; ++i) {
            std::cout << "  i=" << i
                      << "  x="   << static_cast<float>(x_h[i])
                      << "  ref=" << static_cast<float>(y_ref[i])
                      << "  dev=" << static_cast<float>(y_dev[i]) << std::endl;
        }
    }
    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_t);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
