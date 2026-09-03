// Host for diag_gelu_kernel: ./ascend_diag_gelu <N> <mode>
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

extern "C" int aclrtlaunch_diag_gelu_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const uint32_t N    = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8u;
    const uint32_t mode = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 0u;
    check("aclInit", aclInit(nullptr));
    check("aclrtSetDevice", aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N);
    for (uint32_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
    }

    const size_t nbytes = size_t(N) * sizeof(half_t);
    void* d_x = nullptr; check("aclrtMalloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("aclrtMalloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t tiling[2] = {N, mode};
    void* d_tile = nullptr;
    const size_t tsz = sizeof(tiling);
    check("aclrtMalloc tiling", aclrtMalloc(&d_tile, tsz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMemcpy H2D tiling", aclrtMemcpy(d_tile, tsz, tiling, tsz, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D x", aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    static constexpr uint32_t TILE = 256u;
    const uint32_t numBlocks = std::min<uint32_t>(32768u, (N + TILE - 1u) / TILE);
    const auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_diag_gelu_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    if (rc != 0) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<half_t> y_dev(N);
    check("aclrtMemcpy D2H y", aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "=== diag_gelu mode=" << mode << "  N=" << N
              << "  ms=" << ms << " ===" << std::endl;
    std::cout.precision(7);
    for (uint32_t i = 0; i < std::min<uint32_t>(N, 16u); ++i) {
        std::cout << "  i=" << i
                  << "  x="   << static_cast<float>(x_h[i])
                  << "  dev=" << static_cast<float>(y_dev[i])
                  << std::endl;
    }
    std::cout.precision(6);

    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
