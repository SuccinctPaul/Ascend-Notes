// =============================================================================
// RoPE host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 用法:
//   ./ascend_rope <rows> <D>
//     默认 rows=16, D=128 (production kernel: rope_kernel)
//
// 流程 (对齐 rmsnorm_host.cpp 模板):
//   1. 初始化 ACL 运行时
//   2. host 生成 fp16 随机 q/k (rows × D); positions = 0..rows-1
//   3. CPU 侧预计算 cos/sin 表 (rows × D/2, fp32 → fp16) + fp32 参考旋转
//   4. H2D q/k/cos/sin, 下发 tiling ([num_rows, D])
//   5. aclrtlaunch_rope_kernel 启动
//   6. D2H q_out/k_out 回 host, 与 reference 做 allclose + 保范数校验
//   7. 打印 PASS/FAIL + 最大误差 + 耗时
//
// 容差: fp16 级 atol=5e-3, rtol=5e-3 (cos/sin 表本身也是 fp16, 误差略宽)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <string>
#include "acl/acl.h"

using half_t = __fp16;

// ---- ascendc.cmake 生成的 launch 函数 ----
extern "C" int aclrtlaunch_rope_kernel(uint32_t numBlocks, aclrtStream stream,
                                       void* q, void* k, void* cos_table, void* sin_table,
                                       void* q_out, void* k_out,
                                       void* workspace, void* tiling);

// ---- host fp32 参考: 交错配对旋转 ----
static void rope_host_ref(const float* x_fp32, const float* cos_t, const float* sin_t,
                          float* y_fp32, uint32_t rows, uint32_t D) {
    const uint32_t HALF = D / 2u;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t a = 0; a < HALF; ++a) {
            const float x1 = x_fp32[size_t(r) * D + 2u * a];
            const float x2 = x_fp32[size_t(r) * D + 2u * a + 1u];
            const float c  = cos_t[size_t(r) * HALF + a];
            const float s  = sin_t[size_t(r) * HALF + a];
            y_fp32[size_t(r) * D + 2u * a]     = x1 * c - x2 * s;
            y_fp32[size_t(r) * D + 2u * a + 1u] = x1 * s + x2 * c;
        }
    }
}

static void check(const char* where, aclError err) {
    if (err != ACL_ERROR_NONE) {
        std::cerr << "[ACL ERROR] " << where << ": code=" << int(err) << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // ---- 解析命令行 ----
    const uint32_t rows = (argc > 1) ? uint32_t(std::stoul(argv[1])) : 16u;
    const uint32_t D    = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 128u;
    if (D % 2 != 0) {
        std::cerr << "D 必须为偶数 (RoPE 语义), 得到 " << D << std::endl;
        return 1;
    }
    const uint32_t HALF = D / 2u;

    // ---- 1. 初始化 ACL ----
    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- 2. host: 生成 fp16 随机 q/k + positions = 0..rows-1 ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    const size_t N = size_t(rows) * D;
    std::vector<float> q_fp32(N), k_fp32(N);
    std::vector<half_t> q_h(N), k_h(N);
    for (size_t i = 0; i < N; ++i) {
        q_fp32[i] = dist(rng);
        k_fp32[i] = dist(rng);
        q_h[i] = static_cast<half_t>(q_fp32[i]);
        k_h[i] = static_cast<half_t>(k_fp32[i]);
    }

    // ---- 3. 预计算 cos/sin 表 (θ_a = base^(-2a/d), base=10000; fp32 → fp16) ----
    const double kBase = 10000.0;
    std::vector<float> cos_t_fp32(size_t(rows) * HALF), sin_t_fp32(size_t(rows) * HALF);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t a = 0; a < HALF; ++a) {
            const double theta = double(r) * std::pow(kBase, -2.0 * double(a) / double(D));
            cos_t_fp32[size_t(r) * HALF + a] = float(std::cos(theta));
            sin_t_fp32[size_t(r) * HALF + a] = float(std::sin(theta));
        }
    }
    std::vector<half_t> cos_h(size_t(rows) * HALF), sin_h(size_t(rows) * HALF);
    for (size_t i = 0; i < cos_h.size(); ++i) {
        cos_h[i] = static_cast<half_t>(cos_t_fp32[i]);
        sin_h[i] = static_cast<half_t>(sin_t_fp32[i]);
    }
    // 参考计算用 fp16 cast back 的表 (与 device 实际数据一致)
    for (size_t i = 0; i < cos_h.size(); ++i) {
        cos_t_fp32[i] = static_cast<float>(cos_h[i]);
        sin_t_fp32[i] = static_cast<float>(sin_h[i]);
    }

    // ---- 参考输出 (fp32) ----
    std::vector<float>  q_ref_fp32(N), k_ref_fp32(N);
    std::vector<half_t> q_ref_h(N), k_ref_h(N);
    for (size_t i = 0; i < N; ++i) { q_fp32[i] = static_cast<float>(q_h[i]); k_fp32[i] = static_cast<float>(k_h[i]); }
    rope_host_ref(q_fp32.data(), cos_t_fp32.data(), sin_t_fp32.data(), q_ref_fp32.data(), rows, D);
    rope_host_ref(k_fp32.data(), cos_t_fp32.data(), sin_t_fp32.data(), k_ref_fp32.data(), rows, D);
    for (size_t i = 0; i < N; ++i) {
        q_ref_h[i] = static_cast<half_t>(q_ref_fp32[i]);
        k_ref_h[i] = static_cast<half_t>(k_ref_fp32[i]);
    }

    // ---- 4. device buffer ----
    const size_t nbytes = N * sizeof(half_t);
    const size_t cbytes = size_t(rows) * HALF * sizeof(half_t);
    void *d_q = nullptr, *d_k = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    void *d_qo = nullptr, *d_ko = nullptr;
    check("aclrtMalloc q",  aclrtMalloc(&d_q, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMalloc k",  aclrtMalloc(&d_k, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMalloc cos", aclrtMalloc(&d_cos, cbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMalloc sin", aclrtMalloc(&d_sin, cbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMalloc qo", aclrtMalloc(&d_qo, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMalloc ko", aclrtMalloc(&d_ko, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // Tiling: [num_rows, D] (16 字节; 本 kernel 不需要浮点常数)
    struct alignas(8) RopeTiling {
        uint32_t num_rows;  // [0]
        uint32_t D;         // [1]
        uint32_t pad;       // [2]
        uint32_t pad2;      // [3]
    };
    RopeTiling t{};
    t.num_rows = rows;
    t.D = D;
    t.pad = 0u; t.pad2 = 0u;
    const size_t tiling_sz = sizeof(t);
    void* d_tile = nullptr;
    check("aclrtMalloc tiling",
          aclrtMalloc(&d_tile, tiling_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMemcpy H2D tiling",
          aclrtMemcpy(d_tile, tiling_sz, &t, tiling_sz, ACL_MEMCPY_HOST_TO_DEVICE));

    check("aclrtMemcpy H2D q",   aclrtMemcpy(d_q, nbytes, q_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D k",   aclrtMemcpy(d_k, nbytes, k_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D cos", aclrtMemcpy(d_cos, cbytes, cos_h.data(), cbytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D sin", aclrtMemcpy(d_sin, cbytes, sin_h.data(), cbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 5. 下发 kernel ----
    const uint32_t numBlocks = 1u;
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_rope_kernel(numBlocks, stream, d_q, d_k, d_cos, d_sin,
                                           d_qo, d_ko, nullptr, d_tile);
    if (rc != 0) {
        std::cerr << "aclrtlaunch_rope_kernel returned " << rc << "\n";
        return 2;
    }
    check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- 6. D2H 取结果 ----
    std::vector<half_t> qo_dev(N), ko_dev(N);
    check("aclrtMemcpy D2H qo", aclrtMemcpy(qo_dev.data(), nbytes, d_qo, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));
    check("aclrtMemcpy D2H ko", aclrtMemcpy(ko_dev.data(), nbytes, d_ko, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 7. 校验: allclose (q 与 k) + 每对范数守恒 ----
    float max_abs = 0.0f;
    size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    auto verify = [&](const std::vector<half_t>& ref_h, const std::vector<half_t>& dev_h) {
        for (size_t i = 0; i < N; ++i) {
            const float a = static_cast<float>(ref_h[i]);
            const float b = static_cast<float>(dev_h[i]);
            const float err = std::fabs(a - b);
            if (err > max_abs) max_abs = err;
            const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
            if (err / denom > 1.0f) ++bad;
        }
    };
    verify(q_ref_h, qo_dev);
    verify(k_ref_h, ko_dev);

    // 保范数: 每对 (y[2a], y[2a+1]) 的范数 == 输入对应对的范数 (|cos/sin 表误差| ≤ ε)
    float norm_drift = 0.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t a = 0; a < HALF; ++a) {
            const size_t i1 = size_t(r) * D + 2ull * a;
            const size_t i2 = i1 + 1;
            const float nin  = std::sqrt(float(q_h[i1]) * float(q_h[i1]) + float(q_h[i2]) * float(q_h[i2]));
            const float nout = std::sqrt(float(qo_dev[i1]) * float(qo_dev[i1]) + float(qo_dev[i2]) * float(qo_dev[i2]));
            norm_drift = std::fmax(norm_drift, std::fabs(nin - nout));
        }
    }

    const bool pass = (bad == 0) && (norm_drift < 5e-2f);
    std::cout << "=== Ascend C RoPE ===" << std::endl
              << "rows          = " << rows << std::endl
              << "D             = " << D << " (HALF=" << HALF << ")" << std::endl
              << "kernel ms     = " << ms << " (含同步, 仅粗测)" << std::endl
              << "max_abs_err   = " << max_abs << std::endl
              << "bad_elements  = " << bad << " / " << (2u * N) << std::endl
              << "norm_drift    = " << norm_drift << " (per-pair |n_in - n_out| max)" << std::endl
              << "result        = " << (pass ? "PASS" : "FAIL") << std::endl;

    // ---- 8. 清理 ----
    aclrtFree(d_q); aclrtFree(d_k); aclrtFree(d_cos); aclrtFree(d_sin);
    aclrtFree(d_qo); aclrtFree(d_ko); aclrtFree(d_tile);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
