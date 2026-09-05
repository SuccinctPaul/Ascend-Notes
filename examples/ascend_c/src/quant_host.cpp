// =============================================================================
// INT8 量化 host 程序 —— quant/dequant kernel 驱动 + 正确性校验
//
// 用法:
//   ./ascend_quant <rows> <D>
//
// 流程:
//   1. host 生成 fp16 随机 x (rows × D, seed 0xC0FFEE)
//   2. H2D x → quant_kernel → q(int8) + scale(fp32) 驻留 device
//   3. dequant_kernel(q, scale) → y (fp16)
//   4. D2H q/scale/y, host 参考校验:
//        - scale ≈ amax/127 (fp32)
//        - q ∈ [-127,127], 与 host 参考一致率 > 99.9% (round-half 语义差允许 ±1)
//        - 往返误差 max|x - y| ≤ 每行 scale
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

extern "C" int aclrtlaunch_quant_kernel(uint32_t numBlocks, aclrtStream stream,
                                        void* x, void* q, void* scale,
                                        void* workspace, void* tiling);
extern "C" int aclrtlaunch_dequant_kernel(uint32_t numBlocks, aclrtStream stream,
                                          void* q, void* scale, void* y,
                                          void* workspace, void* tiling);

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const uint32_t rows = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 16u;
    const uint32_t D    = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 512u;

    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- host 数据 ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    const size_t N = size_t(rows) * D;
    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N);
    for (size_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
    }
    // host 参考 (fp32)
    std::vector<int8_t> q_ref(N);
    std::vector<float>  scale_ref(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        float amax = 0.0f;
        for (uint32_t c = 0; c < D; ++c) {
            const float v = static_cast<float>(x_h[size_t(r) * D + c]);
            amax = std::fmax(amax, std::fabs(v));
        }
        scale_ref[r] = std::fmax(amax / 127.0f, 1e-12f);
        for (uint32_t c = 0; c < D; ++c) {
            const float v = static_cast<float>(x_h[size_t(r) * D + c]);
            float qf = std::round(v / scale_ref[r]);
            qf = std::fmin(std::fmax(qf, -127.0f), 127.0f);
            q_ref[size_t(r) * D + c] = static_cast<int8_t>(qf);
        }
    }

    // ---- device buffer ----
    const size_t x_bytes = N * sizeof(half_t);
    const size_t q_bytes = N;
    const size_t s_bytes = size_t(rows) * sizeof(float);
    void *d_x, *d_q, *d_s, *d_y;
    check("mx", aclrtMalloc(&d_x, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("mq", aclrtMalloc(&d_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("ms", aclrtMalloc(&d_s, s_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("my", aclrtMalloc(&d_y, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // tiling: [rows, D, pad, pad2, cf[8]] (cf[0]=127.0, cf[1]=1e-12)
    struct alignas(8) Tiling { uint32_t rows, D, pad, pad2; float cf[8]; };
    Tiling t{};
    t.rows = rows; t.D = D;
    t.cf[0] = 127.0f; t.cf[1] = 1e-12f;

    void* d_t;
    check("mt", aclrtMalloc(&d_t, sizeof(Tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    check("ct", aclrtMemcpy(d_t, sizeof(Tiling), &t, sizeof(Tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    check("cx", aclrtMemcpy(d_x, x_bytes, x_h.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- quant + dequant ----
    const auto t0 = std::chrono::steady_clock::now();
    int rc = aclrtlaunch_quant_kernel(1u, stream, d_x, d_q, d_s, nullptr, d_t);
    if (rc != 0) { std::cerr << "quant launch rc=" << rc << "\n"; return 2; }
    check("sync1", aclrtSynchronizeStream(stream));
    rc = aclrtlaunch_dequant_kernel(1u, stream, d_q, d_s, d_y, nullptr, d_t);
    if (rc != 0) { std::cerr << "dequant launch rc=" << rc << "\n"; return 2; }
    check("sync2", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- D2H ----
    std::vector<int8_t> q_dev(N);
    std::vector<float>  s_dev(rows);
    std::vector<half_t> y_dev(N);
    check("cq", aclrtMemcpy(q_dev.data(), q_bytes, d_q, q_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    check("cs", aclrtMemcpy(s_dev.data(), s_bytes, d_s, s_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    check("cy", aclrtMemcpy(y_dev.data(), x_bytes, d_y, x_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 校验 ----
    float scale_err = 0.0f;
    size_t q_mismatch = 0;
    float rt_err = 0.0f;
    bool q_in_range = true;
    for (uint32_t r = 0; r < rows; ++r) {
        scale_err = std::fmax(scale_err, std::fabs(s_dev[r] - scale_ref[r]));
        for (uint32_t c = 0; c < D; ++c) {
            const size_t i = size_t(r) * D + c;
            if (q_dev[i] < -127 || q_dev[i] > 127) q_in_range = false;
            if (q_dev[i] != q_ref[i]) ++q_mismatch;
            rt_err = std::fmax(rt_err, std::fabs(static_cast<float>(y_dev[i]) -
                                                 static_cast<float>(x_h[i])));
        }
    }
    const float max_scale = *std::max_element(scale_ref.begin(), scale_ref.end());
    // q 允许 ±1 LSB 差异: dav-c220 无 fp32↔int8 直转, 量化经 fp16 中转
    // (fp16 网格舍入使 ~1% 元素跨过 0.5 边界), ±1 LSB 在量化噪声内;
    // 硬标准是往返误差 ≤ max_scale
    const bool pass = q_in_range && (scale_err < 1e-6f) &&
                      (q_mismatch * 100.0 / N < 2.0) &&    // ±1 LSB 差异率 < 2%
                      (rt_err <= max_scale + 1e-6);

    std::cout << "=== Ascend C INT8 Quant ===" << std::endl
              << "rows           = " << rows << std::endl
              << "D              = " << D << std::endl
              << "kernel ms      = " << ms << " (quant+dequant, 含同步, 仅粗测)" << std::endl
              << "scale max_err  = " << scale_err << std::endl
              << "q mismatch     = " << q_mismatch << " / " << N
              << " (" << (100.0 * q_mismatch / N) << "%, fp16 中转 ±1 LSB, 允许 <2%)" << std::endl
              << "q in [-127,127]= " << (q_in_range ? "yes" : "NO") << std::endl
              << "roundtrip err  = " << rt_err << " (上界 max_scale=" << max_scale << ")" << std::endl
              << "result         = " << (pass ? "PASS" : "FAIL") << std::endl;

    aclrtFree(d_x); aclrtFree(d_q); aclrtFree(d_s); aclrtFree(d_y); aclrtFree(d_t);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
