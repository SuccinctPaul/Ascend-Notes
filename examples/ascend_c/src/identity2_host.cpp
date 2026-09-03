// Host: ./ascend_identity2 <N> <mode>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include "acl/acl.h"
using half_t = __fp16;
extern "C" int aclrtlaunch_identity2_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
static void check(const char* w, aclError e) {
    if (e) { std::cerr << "[ACL] " << w << " code=" << int(e) << "\n"; std::exit(1); }
}
int main(int argc, char** argv) {
    const uint32_t N    = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8u;
    const uint32_t mode = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 0u;
    check("aclInit", aclInit(nullptr));
    check("setDev", aclrtSetDevice(0));
    aclrtStream s = nullptr;
    check("cs", aclrtCreateStream(&s));

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<half_t> x_h(N);
    for (uint32_t i = 0; i < N; ++i) x_h[i] = static_cast<half_t>(dist(rng));

    const size_t nb = N * sizeof(half_t);
    void *dx, *dy, *dt;
    check("mx", aclrtMalloc(&dx, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    check("my", aclrtMalloc(&dy, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mt", aclrtMalloc(&dt, 8u, ACL_MEM_MALLOC_HUGE_FIRST));
    check("hx", aclrtMemcpy(dx, nb, x_h.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    uint32_t t[2] = {N, mode};
    check("ht", aclrtMemcpy(dt, 8u, t, 8u, ACL_MEMCPY_HOST_TO_DEVICE));

    const uint32_t nbk = std::min<uint32_t>(32768u, mode==3u? 1u : N);
    auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_identity2_kernel(nbk, s, dx, dy, nullptr, dt);
    if (rc) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(s));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
    std::vector<half_t> y_dev(N);
    check("hy", aclrtMemcpy(y_dev.data(), nb, dy, nb, ACL_MEMCPY_DEVICE_TO_HOST));

    float mx = 0; size_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float a = static_cast<float>(x_h[i]);
        float b = static_cast<float>(y_dev[i]);
        float e = std::fabs(a-b);
        if (e > mx) mx = e;
        if (e > 1e-5f) ++bad;
    }
    std::cout << "=== IDENTITY-v2 mode=" << mode << "  N=" << N
              << "  nbk=" << nbk << "  ms=" << ms << "\n"
              << "  max_abs_err=" << mx << "  bad=" << bad << "/" << N
              << "  => " << (bad?"FAIL":"PASS") << "\n";
    if (N <= 32u) {
        std::cout.precision(7);
        for (uint32_t i = 0; i < N; ++i) {
            float a = static_cast<float>(x_h[i]);
            float b = static_cast<float>(y_dev[i]);
            std::cout << "  i=" << i
                      << " x="  << a
                      << " dev="<< b
                      << (std::fabs(a-b)>1e-5f?" WRONG":"")
                      << "\n";
        }
        std::cout.precision(6);
    }
    aclrtFree(dx); aclrtFree(dy); aclrtFree(dt);
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return bad?3:0;
}
