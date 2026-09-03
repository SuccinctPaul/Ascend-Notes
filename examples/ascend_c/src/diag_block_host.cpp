// Host: ./ascend_diag_block <N> <numBlocks>
#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include "acl/acl.h"
using half_t = __fp16;
extern "C" int aclrtlaunch_diag_block_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
static void check(const char* w, aclError e) {
    if (e) { std::cerr << "[ACL] " << w << " code=" << int(e) << "\n"; std::exit(1); }
}

// SENTINEL: 未被任何 block 写的 diag 保持 INT32_MIN
constexpr int32_t SENT = INT32_MIN;

int main(int argc, char** argv) {
    const uint32_t N       = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 16u;
    const uint32_t NBK     = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 16u;
    const uint32_t MAX_DIAG = std::min<uint32_t>(256u, std::max<uint32_t>(NBK, 1u));

    check("aclInit", aclInit(nullptr));
    check("setDev", aclrtSetDevice(0));
    aclrtStream s = nullptr; check("cs", aclrtCreateStream(&s));

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<half_t> x_h(N);
    for (uint32_t i = 0; i < N; ++i) x_h[i] = half_t(dist(rng));
    std::vector<half_t> y_init(N, half_t(-999.0f));  // sentinel

    const size_t nb       = N * sizeof(half_t);
    const size_t diag_sz  = (size_t)MAX_DIAG * 2u * sizeof(int32_t);
    void *dx, *dy, *dt, *dws;
    check("mx", aclrtMalloc(&dx, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    check("my", aclrtMalloc(&dy, nb, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mt", aclrtMalloc(&dt, 8u, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mws", aclrtMalloc(&dws, diag_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("hx", aclrtMemcpy(dx, nb, x_h.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    check("hy_init", aclrtMemcpy(dy, nb, y_init.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int32_t> diag_h(MAX_DIAG * 2u, SENT);
    check("hws_init", aclrtMemcpy(dws, diag_sz, diag_h.data(), diag_sz, ACL_MEMCPY_HOST_TO_DEVICE));
    uint32_t tile[2] = {N, MAX_DIAG};
    check("ht", aclrtMemcpy(dt, 8u, tile, 8u, ACL_MEMCPY_HOST_TO_DEVICE));

    auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_diag_block_kernel(NBK, s, dx, dy, dws, dt);
    if (rc) { std::cerr << "launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(s));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();

    std::vector<half_t> y_dev(N);
    check("hy", aclrtMemcpy(y_dev.data(), nb, dy, nb, ACL_MEMCPY_DEVICE_TO_HOST));
    check("hws", aclrtMemcpy(diag_h.data(), diag_sz, dws, diag_sz, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "=== diag_block_kernel: N=" << N << "  requested_blocks=" << NBK
              << "  ms=" << ms << "\n";
    std::cout << "  -- Block diagnostics (slot i: (bid, total)) -- SENTINEL=INT32_MIN == block never ran\n";
    int32_t seen_min = SENT, seen_max = SENT;
    uint32_t ran_count = 0;
    for (uint32_t i = 0; i < MAX_DIAG; ++i) {
        int32_t bid = diag_h[2*i];
        int32_t tot = diag_h[2*i + 1];
        if (bid == SENT) continue;
        if (seen_min == SENT || bid < seen_min) seen_min = bid;
        if (seen_max == SENT || bid > seen_max) seen_max = bid;
        ++ran_count;
        if (ran_count <= 32u || i == MAX_DIAG-1u) {
            std::cout << "    slot[" << i << "]  bid=" << bid << "  tot=" << tot << "\n";
        }
    }
    if (ran_count > 32u) std::cout << "    ...  (only first 32 shown)\n";
    std::cout << "    total_blocks_ran_uniq = " << ran_count
              << "   bid range: [" << (seen_min==SENT?-1:seen_min)
              << ", " << (seen_max==SENT?-1:seen_max) << "]\n";

    // identity 结果
    uint32_t bad_y = 0;
    uint32_t not_written = 0;
    float mx = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float a = static_cast<float>(x_h[i]);
        float b = static_cast<float>(y_dev[i]);
        float e = std::fabs(a-b);
        if (std::fabs(b - (-999.0f)) < 1.0f) ++not_written;
        else if (e > 1e-5f) ++bad_y;
        if (e > mx) mx = e;
    }
    std::cout << "  -- Identity correctness --\n"
              << "    max_abs_err=" << mx << "  bad=" << (bad_y+not_written)
              << "  (not_written=" << not_written << ", wrong=" << bad_y << ") / " << N
              << "  => " << ((bad_y+not_written)?"FAIL":"PASS") << "\n";

    aclrtFree(dx); aclrtFree(dy); aclrtFree(dws); aclrtFree(dt);
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
