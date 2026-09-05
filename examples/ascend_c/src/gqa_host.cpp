// =============================================================================
// GQA 解码注意力 host 程序 —— kernel 驱动 + 正确性校验
//
// 用法:
//   ./ascend_gqa <Hq> <Hkv> <S> <D>
//     默认 Hq=8, Hkv=2, S=256, D=128
//
// 流程: host 生成 q/K/V (fp16, seed 0xC0FFEE) → CPU fp32 参考注意力 →
//       H2D → gqa_kernel → D2H out → allclose 校验 + 耗时
// 容差: fp16 级 atol=5e-3, rtol=5e-3 (softmax 分数经 fp16 scratch 有轻微舍入)
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

extern "C" int aclrtlaunch_gqa_kernel(uint32_t numBlocks, aclrtStream stream,
                                      void* q, void* k, void* v, void* out, void* scratch,
                                      void* workspace, void* tiling);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const uint32_t Hq  = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 8u;
    const uint32_t Hkv = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 2u;
    const uint32_t S   = (argc > 3) ? uint32_t(std::stoul(argv[3])) : 256u;
    const uint32_t D   = (argc > 4) ? uint32_t(std::stoul(argv[4])) : 128u;
    if (Hq % Hkv != 0) { std::cerr << "Hq 必须被 Hkv 整除\n"; return 1; }
    const uint32_t G = Hq / Hkv;

    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- host 数据 ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    const size_t Nq = size_t(Hq) * D;
    const size_t Nkv = size_t(Hkv) * S * D;
    std::vector<half_t> q_h(Nq), k_h(Nkv), v_h(Nkv);
    for (size_t i = 0; i < Nq; ++i) q_h[i] = static_cast<half_t>(dist(rng));
    for (size_t i = 0; i < Nkv; ++i) { k_h[i] = static_cast<half_t>(dist(rng)); v_h[i] = static_cast<half_t>(dist(rng)); }

    // ---- CPU fp32 参考 ----
    std::vector<float> q32(Nq), k32(Nkv), v32(Nkv);
    for (size_t i = 0; i < Nq; ++i) q32[i] = static_cast<float>(q_h[i]);
    for (size_t i = 0; i < Nkv; ++i) { k32[i] = static_cast<float>(k_h[i]); v32[i] = static_cast<float>(v_h[i]); }
    std::vector<half_t> ref_h(Nq);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    for (uint32_t hq = 0; hq < Hq; ++hq) {
        const uint32_t kv = hq / G;
        std::vector<float> sc(S);
        float m = -1e30f;
        for (uint32_t s = 0; s < S; ++s) {
            float acc = 0.0f;
            for (uint32_t d = 0; d < D; ++d)
                acc += q32[size_t(hq) * D + d] * k32[(size_t(kv) * S + s) * D + d];
            sc[s] = acc * inv_sqrt_d;
            m = std::fmax(m, sc[s]);
        }
        float l = 0.0f;
        for (uint32_t s = 0; s < S; ++s) { sc[s] = std::exp(sc[s] - m); l += sc[s]; }
        for (uint32_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < S; ++s)
                acc += sc[s] * v32[(size_t(kv) * S + s) * D + d];
            ref_h[size_t(hq) * D + d] = static_cast<half_t>(acc / l);
        }
    }

    // ---- device buffer ----
    const size_t q_bytes = Nq * 2, kv_bytes = Nkv * 2, sc_bytes = size_t(Hq) * S * 2;
    void *d_q, *d_k, *d_v, *d_o, *d_sc;
    check("mq",  aclrtMalloc(&d_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mk",  aclrtMalloc(&d_k, kv_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mv",  aclrtMalloc(&d_v, kv_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mo",  aclrtMalloc(&d_o, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("msc", aclrtMalloc(&d_sc, sc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // tiling: [Hq, G, S, D, pad] + cf[8] (cf[0]=1/sqrt(D)) = 48 字节
    struct alignas(8) Tiling { uint32_t Hq, G, S, D; float cf[8]; };
    Tiling t{};
    t.Hq = Hq; t.G = G; t.S = S; t.D = D;
    t.cf[0] = inv_sqrt_d;
    void* d_t;
    check("mt", aclrtMalloc(&d_t, sizeof(Tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    check("ct", aclrtMemcpy(d_t, sizeof(Tiling), &t, sizeof(Tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    check("cq", aclrtMemcpy(d_q, q_bytes, q_h.data(), q_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("ck", aclrtMemcpy(d_k, kv_bytes, k_h.data(), kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("cv", aclrtMemcpy(d_v, kv_bytes, v_h.data(), kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- launch ----
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_gqa_kernel(1u, stream, d_q, d_k, d_v, d_o, d_sc, nullptr, d_t);
    if (rc != 0) { std::cerr << "gqa launch rc=" << rc << "\n"; return 2; }
    check("sync", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- D2H + 校验 ----
    std::vector<half_t> out_h(Nq);
    check("co", aclrtMemcpy(out_h.data(), q_bytes, d_o, q_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    float max_abs = 0.0f;
    size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    for (size_t i = 0; i < Nq; ++i) {
        const float a = static_cast<float>(ref_h[i]);
        const float b = static_cast<float>(out_h[i]);
        const float err = std::fabs(a - b);
        if (err > max_abs) max_abs = err;
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }
    const bool pass = (bad == 0);
    std::cout << "=== Ascend C GQA Decode ===" << std::endl
              << "Hq/Hkv        = " << Hq << "/" << Hkv << std::endl
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
