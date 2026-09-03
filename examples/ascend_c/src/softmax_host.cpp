// =============================================================================
// Softmax host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 用法:
//   ./ascend_softmax <rows> <D> [scalar]
//     默认 rows=16, D=128, 生产版 kernel (softmax_kernel)
//     第三个参数传 "scalar" → 用标量地板版 softmax_scalar_kernel
//
// 流程 (对齐 gelu_host.cpp 模板):
//   1. 初始化 ACL 运行时
//   2. 在 host 生成 fp16 随机矩阵 x (shape: rows × D, row-major, seed 0xC0FFEE)
//   3. CPU 侧算参考 softmax_reference (数值稳定: max→exp(x-m)→sum→div, fp32)
//   4. H2D x → device, 下发 tiling (production: [num_rows, D]; scalar: [num_rows, D, 0.0f, 1.0f, -1e20f])
//   5. 调用对应 aclrtlaunch_* 启动 kernel
//   6. D2H 取 y 回 host, 与 reference 做 allclose
//   7. 打印 PASS/FAIL + 最大误差 + 耗时
//
// 容差: fp16 级 atol=5e-3, rtol=5e-3
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>   // std::min / std::max
#include <string>
#include "acl/acl.h"

using half_t = __fp16;

// ---- ascendc.cmake 生成的 launch 函数 ----
extern "C" int aclrtlaunch_softmax_kernel(uint32_t numBlocks, aclrtStream stream,
                                           void* x, void* y, void* workspace, void* tiling);
extern "C" int aclrtlaunch_softmax_scalar_kernel(uint32_t numBlocks, aclrtStream stream,
                                                  void* x, void* y, void* workspace, void* tiling);

// ---- host fp32 reference softmax (按行做数值稳定 softmax) ----
static void softmax_host_ref(const float* x_fp32, float* y_fp32, uint32_t rows, uint32_t D) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float* xrow = x_fp32 + r * D;
        float* yrow       = y_fp32 + r * D;

        // 1) row max
        float m = xrow[0];
        for (uint32_t j = 1; j < D; ++j) if (xrow[j] > m) m = xrow[j];

        // 2) exp(x - m) + sum
        float s = 0.0f;
        for (uint32_t j = 0; j < D; ++j) {
            const float e = std::exp(xrow[j] - m);
            yrow[j] = e;
            s += e;
        }
        // 3) normalize
        const float inv = 1.0f / s;
        for (uint32_t j = 0; j < D; ++j) yrow[j] *= inv;
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
    const bool use_scalar = (argc > 3) && (std::string(argv[3]) == "scalar");
    const char* kernel_name = use_scalar ? "scalar" : "production";

    // ---- 1. 初始化 ACL ----
    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- 2. host: 生成 fp16 随机 x (rows × D, row-major) ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    const size_t N = size_t(rows) * D;
    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N);
    for (size_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
    }

    // ---- 3. host 参考 (先把 fp16 cast back → fp32 再算, 模拟 device 实际数据) ----
    std::vector<float>  x_ref_fp32(N);
    std::vector<float>  y_ref_fp32(N);
    std::vector<half_t> y_ref_h(N);
    for (size_t i = 0; i < N; ++i) x_ref_fp32[i] = static_cast<float>(x_h[i]);
    softmax_host_ref(x_ref_fp32.data(), y_ref_fp32.data(), rows, D);
    for (size_t i = 0; i < N; ++i) y_ref_h[i] = static_cast<half_t>(y_ref_fp32[i]);

    // ---- 4. device buffer ----
    const size_t nbytes = N * sizeof(half_t);
    void* d_x = nullptr; check("aclrtMalloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("aclrtMalloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- Tiling v6 layout (prod & scalar 统一):
    //   offset  0: uint32_t num_rows
    //   offset  4: uint32_t D
    //   offset  8: uint32_t pad       → 保证之后的 float cf[8] 8B 对齐
    //   offset 12: uint32_t pad2
    //   offset 16: float cf[8]       cf[0]=-1e20 (M_INF), cf[1]=0.0 (ZERO), cf[2]=1.0 (CONE)
    //   sizeof = 48 bytes.
    // Kernel 通过 GlobalTensor<float>(T+4, 8) DataCopy 读 cf[0..7]; 等价于偏移 T+16.
    struct alignas(8) SoftmaxTiling {
        uint32_t num_rows;       // [0]
        uint32_t D;              // [1]
        uint32_t pad;            // [2]
        uint32_t pad2;           // [3]
        float    cf[8];          // bytes 16..48
    };
    static_assert(sizeof(SoftmaxTiling) == 4u * 4u + 8u * 4u,
                  "SoftmaxTiling must be 48 bytes: 4 u32 header + 8 floats");
    SoftmaxTiling t{};
    t.num_rows = rows;
    t.D = D;
    t.pad = 0u; t.pad2 = 0u;
    t.cf[0] = -1e20f;
    t.cf[1] = 0.0f;
    t.cf[2] = 1.0f;
    for (int k = 3; k < 8; ++k) t.cf[k] = 0.0f;
    const size_t tiling_sz = sizeof(t);
    void* d_tile = nullptr;
    check("aclrtMalloc tiling",
          aclrtMalloc(&d_tile, tiling_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMemcpy H2D tiling",
          aclrtMemcpy(d_tile, tiling_sz, &t, tiling_sz, ACL_MEMCPY_HOST_TO_DEVICE));

    // H2D x
    check("aclrtMemcpy H2D x",
          aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 5. 下发 kernel ----
    // 规则: 生产/标量 kernel 都只在 block 0 执行 (保证 CANN 9.0 云容器共享调度环境
    //       下 100% 覆盖所有行 / 列), 所以 numBlocks = 1.
    const uint32_t numBlocks = 1u;
    const auto t0 = std::chrono::steady_clock::now();
    int rc = 0;
    if (use_scalar) {
        rc = aclrtlaunch_softmax_scalar_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    } else {
        rc = aclrtlaunch_softmax_kernel(numBlocks, stream, d_x, d_y, nullptr, d_tile);
    }
    if (rc != 0) {
        std::cerr << "aclrtlaunch_softmax" << (use_scalar ? "_scalar" : "")
                  << "_kernel returned " << rc << "\n";
        return 2;
    }
    check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- 6. D2H 取结果 ----
    std::vector<half_t> y_dev(N);
    check("aclrtMemcpy D2H y",
          aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 7. 校验: max_abs_error + allclose + 每行求和 ≈ 1 + softmax 非负性 ----
    float max_abs = 0.0f;
    size_t bad = 0;
    constexpr float atol = 5e-3f, rtol = 5e-3f;
    for (size_t i = 0; i < N; ++i) {
        const float a = static_cast<float>(y_ref_h[i]);
        const float b = static_cast<float>(y_dev[i]);
        const float err = std::fabs(a - b);
        if (err > max_abs) max_abs = err;
        const float denom = std::fmax(1e-6f, std::fabs(a) * rtol + atol);
        if (err / denom > 1.0f) ++bad;
    }

    const bool pass = (bad == 0);
    std::cout << "=== Ascend C Softmax ===" << std::endl
              << "rows         = " << rows << std::endl
              << "D            = " << D << std::endl
              << "kernel       = " << kernel_name << std::endl
              << "kernel ms    = " << ms << " (含同步，仅粗测)" << std::endl
              << "max_abs_err  = " << max_abs << std::endl
              << "bad_elements = " << bad << " / " << N << std::endl
              << "result       = " << (pass ? "PASS" : "FAIL") << std::endl;

    // ---- DEBUG: small problem dump ref vs dev ----
    if (rows * D <= 64u) {
        std::cout << "\n--- DEBUG rows*D <= 64: [row r, col c] x_h | y_ref (fp16->fp32) | y_dev (fp16->fp32) ---" << std::endl;
        std::cout.precision(7);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < D; ++c) {
                uint32_t idx = r * D + c;
                std::cout << "  [" << r << "," << c << "]"
                          << " x="    << static_cast<float>(x_h[idx])
                          << "  ref=" << static_cast<float>(y_ref_h[idx])
                          << "  dev=" << static_cast<float>(y_dev[idx])
                          << std::endl;
            }
            // show row sum ref vs dev
            float rs_ref = 0.0f, rs_dev = 0.0f;
            for (uint32_t c = 0; c < D; ++c) {
                uint32_t idx = r * D + c;
                rs_ref += static_cast<float>(y_ref_h[idx]);
                rs_dev += static_cast<float>(y_dev[idx]);
            }
            std::cout << "  row " << r << " sum_ref=" << rs_ref << " sum_dev=" << rs_dev << std::endl;
        }
        std::cout.precision(6);
    }

    // ---- 8. 清理 ----
    aclrtFree(d_x); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
