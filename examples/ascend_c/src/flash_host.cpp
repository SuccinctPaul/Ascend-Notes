// =============================================================================
// FlashAttention host 程序 —— kernel 驱动 + 正确性校验
//
// 用法:
//   ./ascend_flash <H> <L> <S> <D>
//     默认 H=2, L=64, S=128, D=64 (教学版串行实现, 规模勿过大)
//
// 流程: host 生成 q/k/v (fp16, seed 0xC0FFEE) → CPU fp32 参考注意力 →
//       H2D → flash_kernel → D2H out → allclose 校验 + 耗时
// 容差: fp16 级 atol=1e-2, rtol=1e-2 (acc 借 fp16 OUT 缓冲累加, 舍入略宽)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include "acl/acl.h"

using half_t = __fp16;

extern "C" int aclrtlaunch_flash_kernel(uint32_t numBlocks, aclrtStream stream,
                                        void* q, void* k, void* v, void* out, void* scratch,
                                        void* workspace, void* tiling);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const uint32_t H = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 2u;
    const uint32_t L = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 64u;
    const uint32_t S = (argc > 3) ? uint32_t(std::stoul(argv[3])) : 128u;
    const uint32_t D = (argc > 4) ? uint32_t(std::stoul(argv[4])) : 64u;

    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- host 数据 ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    const size_t Nq = size_t(H) * L * D;
    const size_t Nkv = size_t(H) * S * D;
    const size_t R = size_t(H) * L;
    std::vector<half_t> q_h(Nq), k_h(Nkv), v_h(Nkv);
    for (size_t i = 0; i < Nq; ++i) q_h[i] = static_cast<half_t>(dist(rng));
    for (size_t i = 0; i < Nkv; ++i) { k_h[i] = static_cast<half_t>(dist(rng)); v_h[i] = static_cast<half_t>(dist(rng)); }

    // ---- CPU fp32 参考 (标准注意力, 与 flash 数学等价) ----
    std::vector<float> q32(Nq), k32(Nkv), v32(Nkv);
    for (size_t i = 0; i < Nq; ++i) q32[i] = static_cast<float>(q_h[i]);
    for (size_t i = 0; i < Nkv; ++i) { k32[i] = static_cast<float>(k_h[i]); v32[i] = static_cast<float>(v_h[i]); }
    std::vector<half_t> ref_h(Nq);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    for (uint32_t h = 0; h < H; ++h) {
        std::vector<float> sc(S);
        for (uint32_t m = 0; m < L; ++m) {
            float mx = -1e30f;
            for (uint32_t s = 0; s < S; ++s) {
                float acc = 0.0f;
                for (uint32_t d = 0; d < D; ++d)
                    acc += q32[(size_t(h) * L + m) * D + d] * k32[(size_t(h) * S + s) * D + d];
                sc[s] = acc * inv_sqrt_d;
                mx = std::fmax(mx, sc[s]);
            }
            float l = 0.0f;
            for (uint32_t s = 0; s < S; ++s) { sc[s] = std::exp(sc[s] - mx); l += sc[s]; }
            for (uint32_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < S; ++s)
                    acc += sc[s] * v32[(size_t(h) * S + s) * D + d];
                ref_h[(size_t(h) * L + m) * D + d] = static_cast<half_t>(acc / l);
            }
        }
    }

    // ---- device buffer ----
    const size_t q_bytes = Nq * 2, kv_bytes = Nkv * 2, sc_bytes = R * S * 2;
    void *d_q, *d_k, *d_v, *d_o, *d_sc;
    check("mq",  aclrtMalloc(&d_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mk",  aclrtMalloc(&d_k, kv_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mv",  aclrtMalloc(&d_v, kv_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mo",  aclrtMalloc(&d_o, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("msc", aclrtMalloc(&d_sc, sc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("zo",  aclrtMemset(d_o, q_bytes, 0, q_bytes));

    // tiling: [H, L, S, D] + cf[8] (cf[0]=1/sqrt(D)) = 48 字节
    struct alignas(8) Tiling { uint32_t H, L, S, D; float cf[8]; };
    Tiling t{};
    t.H = H; t.L = L; t.S = S; t.D = D;
    t.cf[0] = inv_sqrt_d;
    void* d_t;
    check("mt", aclrtMalloc(&d_t, sizeof(Tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    check("ct", aclrtMemcpy(d_t, sizeof(Tiling), &t, sizeof(Tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    check("cq", aclrtMemcpy(d_q, q_bytes, q_h.data(), q_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("ck", aclrtMemcpy(d_k, kv_bytes, k_h.data(), kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("cv", aclrtMemcpy(d_v, kv_bytes, v_h.data(), kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- launch ----
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_flash_kernel(1u, stream, d_q, d_k, d_v, d_o, d_sc, nullptr, d_t);
    if (rc != 0) { std::cerr << "flash launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- D2H + 校验 ----
    std::vector<half_t> out_h(Nq);
    check("co", aclrtMemcpy(out_h.data(), q_bytes, d_o, q_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    float max_abs = 0.0f;
    size_t bad = 0;
    constexpr float atol = 1e-2f, rtol = 1e-2f;
    for (size_t i = 0; i < Nq; ++i) {
        const float a = static_cast<float>(ref_h[i]);
        const float b = static_cast<float>(out_h[i]);
        const float err = std::fabs(a - b);
        if (err > max_abs) max_abs = err;
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }
    const bool pass = (bad == 0);
    std::cout << "=== Ascend C FlashAttention ===" << std::endl
              << "H             = " << H << std::endl
              << "L             = " << L << std::endl
              << "S             = " << S << std::endl
              << "D             = " << D << std::endl
              << "kernel ms     = " << ms << " (含同步, 仅粗测)" << std::endl
              << "max_abs_err   = " << max_abs << std::endl
              << "bad_elements  = " << bad << " / " << Nq << std::endl
              << "result        = " << (pass ? "PASS" : "FAIL") << std::endl;

    aclrtFree(d_q); aclrtFree(d_k); aclrtFree(d_v); aclrtFree(d_o); aclrtFree(d_sc); aclrtFree(d_t);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
