// Host for diag2 — y = 0.5 * x
#include <acl/acl.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

using half_t = __fp16;

extern "C" int aclrtlaunch_diag2_kernel(uint32_t blockDim, aclrtStream stream,
    void* x, void* y, void* workspace, void* tiling);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const uint32_t N = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 256u;
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> U(-3.0f, 3.0f);

    aclInit(nullptr);
    check("setDev", aclrtSetDevice(0));
    aclrtContext ctx = nullptr; aclrtCreateContext(&ctx, 0);
    aclrtStream stream = nullptr; aclrtCreateStream(&stream);

    std::vector<half_t> x_h(N);
    std::vector<float> y_ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        float xv = U(rng);
        x_h[i] = static_cast<half_t>(xv);
        y_ref[i] = xv * 0.5f;
    }
    const size_t nb = size_t(N) * sizeof(half_t);
    void* dx = nullptr; check("mX", aclrtMalloc(&dx, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    void* dy = nullptr; check("mY", aclrtMalloc(&dy, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t tN = N; void* dt = nullptr;
    check("mT", aclrtMalloc(&dt, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2Dx", aclrtMemcpy(dx, nb, x_h.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    check("H2Dt", aclrtMemcpy(dt, sizeof(uint32_t), &tN, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t blocks = std::min<uint32_t>(32768u, (N + 255u) / 256u);
    std::cout << "diag2 (y = 0.5x, LocalTensor+DataCopy+MulV) N=" << N << " blocks=" << blocks << "\n";

    auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_diag2_kernel(blocks, stream, dx, dy, nullptr, dt);
    if (rc) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("kernel ms    = %.6f\n", ms);

    std::vector<half_t> y_h(N);
    check("D2H", aclrtMemcpy(y_h.data(), nb, dy, nb, ACL_MEMCPY_DEVICE_TO_HOST));

    float max_abs = 0;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float y = static_cast<float>(y_h[i]);
        float d = std::abs(y - y_ref[i]);
        if (d > 1e-3f) {
            bad++;
            if (i < 16) printf("  i=%u  x=%+.4f  ref=%+.4f  y=%+.4f  d=%.5f\n", i, (float)x_h[i], y_ref[i], y, d);
        }
        max_abs = std::max(max_abs, d);
    }
    printf("max_abs_err  = %.5g\n", max_abs);
    printf("bad_elements = %u / %u\n", bad, N);
    printf("result       = %s\n", (bad == 0 && max_abs < 1e-3f) ? "PASS" : "FAIL");

    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
