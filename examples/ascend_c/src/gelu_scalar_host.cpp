// =============================================================================
// 教学版 Scalar GELU host 驱动 —— 用于"scalar 地板性能"对照
//
// 与 ascend_gelu 唯一区别:
//   · 调用 aclrtlaunch_gelu_scalar_kernel(...)  而非 gelu_kernel
//   · 注释标明"故意不走 Vector/DataCopy, 仅做正确性+地板性能"
//
// 校验协议与 gelu_host.cpp 完全一致:
//   ref = 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
//   容差 atol=rtol=5e-3 (fp16)
// =============================================================================
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

extern "C" int aclrtlaunch_gelu_scalar_kernel(uint32_t numBlocks, aclrtStream stream,
                                               void* x, void* y, void* workspace, void* tiling);

static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
static constexpr double CUBIC_COEF     = 0.044715;

static float gelu_ref(float xv) {
    double x = xv;
    const double inner = SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x);
    return static_cast<float>(x * 0.5 * (1.0 + std::tanh(inner)));
}
static void check(const char* w, aclError e) {
    if (e != ACL_ERROR_NONE) { std::cerr<<"[ACL ERROR] "<<w<<": code="<<int(e)<<"\n"; std::exit(1); }
}

int main(int argc, char** argv) {
    const uint32_t N = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8192u;

    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("setDev", aclrtSetDevice(devId));
    aclrtStream stream = nullptr; check("createStream", aclrtCreateStream(&stream));

    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N);
    for (uint32_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i]    = static_cast<half_t>(x_fp32[i]);
    }

    std::vector<half_t> y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        const float xv = static_cast<float>(x_h[i]);
        y_ref[i] = static_cast<half_t>(gelu_ref(xv));
    }

    const size_t nb = size_t(N) * sizeof(half_t);
    void* d_x = nullptr; check("mx", aclrtMalloc(&d_x, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("my", aclrtMalloc(&d_y, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t tN = N; void* d_t = nullptr;
    check("mt", aclrtMalloc(&d_t, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2Dx", aclrtMemcpy(d_x, nb, x_h.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    check("H2Dt", aclrtMemcpy(d_t, sizeof(uint32_t), &tN, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    // scalar 版是逐元素 grid-stride, 单元素迭代成本高,
    // 这里把核数塞满 (最多 32768 并行块, 与生产版一致)
    static constexpr uint32_t BLK_PER_TILE = 256u;
    const uint32_t numBlocks = std::min<uint32_t>(
        32768u,
        (N + BLK_PER_TILE - 1u) / BLK_PER_TILE
    );

    std::cout << "=== Ascend C GELU (SCALAR TEACHING VER, 地板性能) ===" << std::endl
              << "N       = " << N << "\n"
              << "blocks  = " << numBlocks << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_gelu_scalar_kernel(numBlocks, stream, d_x, d_y, nullptr, d_t);
    if (rc) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<half_t> y_dev(N);
    check("D2H", aclrtMemcpy(y_dev.data(), nb, d_y, nb, ACL_MEMCPY_DEVICE_TO_HOST));

    float max_abs = 0.0f; size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    for (uint32_t i = 0; i < N; ++i) {
        const float a = static_cast<float>(y_ref[i]);
        const float b = static_cast<float>(y_dev[i]);
        const float err = std::fabs(a - b);
        max_abs = std::max(max_abs, err);
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }
    const bool pass = (bad == 0);
    std::cout << "kernel ms    = " << ms << "\n"
              << "max_abs_err  = " << max_abs << "\n"
              << "bad_elements = " << bad << " / " << N << "\n"
              << "result       = " << (pass ? "PASS" : "FAIL") << "\n";

    // 地板性能指标 (粗估)
    const double bytes = 2.0 * double(N) * sizeof(half_t);  // 读+写
    const double secs  = ms * 1e-3;
    const double GBps  = bytes / secs / 1e9;
    const double HBM_THEORY = 1228.8;  // 910B2 理论 HBM (GB/s)
    std::cout << "est. GB/s    = " << GBps
              << "   (HBM util ~" << (100.0 * GBps / HBM_THEORY) << "%)\n";

    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_t);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
