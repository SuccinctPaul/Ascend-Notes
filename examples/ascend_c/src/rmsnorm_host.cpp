// =============================================================================
// RMSNorm host 程序 —— Ascend C kernel 的 host 侧驱动 + 正确性校验
//
// 用法:
//   ./ascend_rmsnorm <rows> <D>
//     默认 rows=16, D=512 (production kernel: rmsnorm_kernel)
//
// 流程 (对齐 softmax_host.cpp 模板):
//   1. 初始化 ACL 运行时
//   2. host 生成 fp16 随机 x (rows × D) + gamma (D, uniform 0.5~2.0, seed 0xC0FFEE)
//   3. CPU 侧算参考 rmsnorm (fp32 归约: Σx² → rms → x·inv_rms·gamma)
//   4. H2D x/gamma, 下发 tiling ([num_rows, D, 0, 0], cf: [eps, 0.0, 1.0])
//   5. aclrtlaunch_rmsnorm_kernel 启动
//   6. D2H y 回 host, 与 reference 做 allclose + 归一化能量校验
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
#include <string>
#include "acl/acl.h"

using half_t = __fp16;

// ---- ascendc.cmake 生成的 launch 函数 ----
extern "C" int aclrtlaunch_rmsnorm_kernel(uint32_t numBlocks, aclrtStream stream,
                                          void* x, void* gamma, void* y,
                                          void* workspace, void* tiling);

// ---- host fp32 reference RMSNorm (fp32 归约 Σx²) ----
static void rmsnorm_host_ref(const float* x_fp32, const float* gamma_fp32,
                             float* y_fp32, uint32_t rows, uint32_t D, float eps) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float* xrow = x_fp32 + r * D;
        float* yrow       = y_fp32 + r * D;
        float sq = 0.0f;
        for (uint32_t j = 0; j < D; ++j) sq += xrow[j] * xrow[j];
        const float inv_rms = 1.0f / std::sqrt(sq / static_cast<float>(D) + eps);
        for (uint32_t j = 0; j < D; ++j) yrow[j] = xrow[j] * inv_rms * gamma_fp32[j];
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
    const uint32_t D    = (argc > 2) ? uint32_t(std::stoul(argv[2])) : 512u;
    constexpr float kEps = 1e-6f;

    // ---- 1. 初始化 ACL ----
    check("aclInit", aclInit(nullptr));
    int32_t devId = 0;
    check("aclrtSetDevice", aclrtSetDevice(devId));
    aclrtStream stream = nullptr;
    check("aclrtCreateStream", aclrtCreateStream(&stream));

    // ---- 2. host: 生成 fp16 随机 x (rows × D) + gamma (D) ----
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::uniform_real_distribution<float> gdist(0.5f, 2.0f);

    const size_t N = size_t(rows) * D;
    std::vector<float> x_fp32(N);
    std::vector<half_t> x_h(N), g_h(D);
    std::vector<float> g_fp32(D);
    for (size_t i = 0; i < N; ++i) {
        x_fp32[i] = dist(rng);
        x_h[i] = static_cast<half_t>(x_fp32[i]);
    }
    for (uint32_t j = 0; j < D; ++j) {
        g_fp32[j] = gdist(rng);
        g_h[j] = static_cast<half_t>(g_fp32[j]);
    }

    // ---- 3. host 参考 (fp16 cast back → fp32 再算, 模拟 device 实际数据) ----
    std::vector<float>  x_ref_fp32(N);
    std::vector<float>  y_ref_fp32(N);
    std::vector<half_t> y_ref_h(N);
    for (size_t i = 0; i < N; ++i) x_ref_fp32[i] = static_cast<float>(x_h[i]);
    for (uint32_t j = 0; j < D; ++j) g_fp32[j] = static_cast<float>(g_h[j]);
    rmsnorm_host_ref(x_ref_fp32.data(), g_fp32.data(), y_ref_fp32.data(), rows, D, kEps);
    for (size_t i = 0; i < N; ++i) y_ref_h[i] = static_cast<half_t>(y_ref_fp32[i]);

    // ---- 4. device buffer ----
    const size_t nbytes = N * sizeof(half_t);
    const size_t gbytes = size_t(D) * sizeof(half_t);
    void* d_x = nullptr; check("aclrtMalloc x", aclrtMalloc(&d_x, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_g = nullptr; check("aclrtMalloc gamma", aclrtMalloc(&d_g, gbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* d_y = nullptr; check("aclrtMalloc y", aclrtMalloc(&d_y, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- Tiling layout (48 字节, 与 softmax 统一):
    //   offset  0: uint32_t num_rows
    //   offset  4: uint32_t D
    //   offset  8: uint32_t pad
    //   offset 12: uint32_t pad2
    //   offset 16: float cf[8]   cf[0]=eps, cf[1]=0.0 (ZERO), cf[2]=1.0 (CONE),
    //                             cf[3]=(float)D (host 转 float, 避免核内 int→float cast)
    struct alignas(8) RmsNormTiling {
        uint32_t num_rows;       // [0]
        uint32_t D;              // [1]
        uint32_t pad;            // [2]
        uint32_t pad2;           // [3]
        float    cf[8];          // bytes 16..48
    };
    static_assert(sizeof(RmsNormTiling) == 4u * 4u + 8u * 4u,
                  "RmsNormTiling must be 48 bytes: 4 u32 header + 8 floats");
    RmsNormTiling t{};
    t.num_rows = rows;
    t.D = D;
    t.pad = 0u; t.pad2 = 0u;
    t.cf[0] = kEps;
    t.cf[1] = 0.0f;
    t.cf[2] = 1.0f;
    t.cf[3] = static_cast<float>(D);
    // cf[4..7] 保持 t{} 的零初始化 —— 注意不要再用清零循环覆盖 cf[3]
    // (2026-09 实测踩坑: cf[3] 赋值后被 "for k=3..8 cf[k]=0" 循环抹零,
    //  DF=0 → inv_rms=0 → 输出全 0, 现象上极像 kernel/缓存问题)
    const size_t tiling_sz = sizeof(t);
    void* d_tile = nullptr;
    check("aclrtMalloc tiling",
          aclrtMalloc(&d_tile, tiling_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    check("aclrtMemcpy H2D tiling",
          aclrtMemcpy(d_tile, tiling_sz, &t, tiling_sz, ACL_MEMCPY_HOST_TO_DEVICE));

    // H2D x / gamma
    check("aclrtMemcpy H2D x",
          aclrtMemcpy(d_x, nbytes, x_h.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
    check("aclrtMemcpy H2D gamma",
          aclrtMemcpy(d_g, gbytes, g_h.data(), gbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 5. 下发 kernel (numBlocks=1, 单核覆盖所有行, 规避多 block 调度遗漏) ----
    const uint32_t numBlocks = 1u;
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = aclrtlaunch_rmsnorm_kernel(numBlocks, stream, d_x, d_g, d_y, nullptr, d_tile);
    if (rc != 0) {
        std::cerr << "aclrtlaunch_rmsnorm_kernel returned " << rc << "\n";
        return 2;
    }
    check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ---- 6. D2H 取结果 ----
    std::vector<half_t> y_dev(N);
    check("aclrtMemcpy D2H y",
          aclrtMemcpy(y_dev.data(), nbytes, d_y, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 7. 校验: allclose + 归一化能量 (y/gamma 均方 ≈ 1) ----
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
    float ms_energy_err = 0.0f;   // max |mean((y/gamma)²) - 1|
    for (uint32_t r = 0; r < rows; ++r) {
        float sq = 0.0f;
        for (uint32_t j = 0; j < D; ++j) {
            const float yv = static_cast<float>(y_dev[size_t(r) * D + j]);
            const float gv = static_cast<float>(g_h[j]);
            sq += (yv / gv) * (yv / gv);
        }
        const float mean_sq = sq / static_cast<float>(D);
        ms_energy_err = std::fmax(ms_energy_err, std::fabs(mean_sq - 1.0f));
    }

    const bool pass = (bad == 0) && (ms_energy_err < 2e-2f);
    std::cout << "=== Ascend C RMSNorm ===" << std::endl
              << "rows            = " << rows << std::endl
              << "D               = " << D << std::endl
              << "kernel ms       = " << ms << " (含同步, 仅粗测)" << std::endl
              << "max_abs_err     = " << max_abs << std::endl
              << "bad_elements    = " << bad << " / " << N << std::endl
              << "energy_err      = " << ms_energy_err << " (|mean((y/gamma)^2)-1| max)" << std::endl
              << "result          = " << (pass ? "PASS" : "FAIL") << std::endl;

    // ---- 8. 清理 ----
    aclrtFree(d_x); aclrtFree(d_g); aclrtFree(d_y); aclrtFree(d_tile);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    return pass ? 0 : 3;
}
