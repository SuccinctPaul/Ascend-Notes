// =============================================================================
// GEMM kernel —— Ascend C (CANN 原生 C++ kernel DSL) 朴素实现
//
// 计算: C = A @ B
//   A ∈ R^{M×K}  (float16, 行主序)
//   B ∈ R^{K×N}  (float16, 行主序)
//   C ∈ R^{M×N}  (float16, 行主序)
//
// 精度策略: fp16 输入 + fp32 累加器 (混合精度, NPU Cube 单元的标准做法)
//
// 入口签名由 CANN 约定:
//   extern "C" __global__ __aicore__ void <kernel_name>(
//       GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
//   - __global__/__aicore__: 表示运行在 AI Core 上
//   - GM_ADDR: 全局内存 (HBM) 指针类型
//   - workspace: 运行时工作区 (本朴素版不用)
//   - tiling: host 下发的标量参数 (这里装 M/K/N)
// =============================================================================

#include "kernel_operator.h"  // Ascend C kernel DSL 头文件 (GlobalTensor / half 等)

using namespace AscendC;

// half: AscendC 提供的 fp16 类型 (对应硬件原生精度)
// 行主序寻址: A[i][j] 在一维数组中为 A[i*K + j]

extern "C" __global__ __aicore__
void gemm_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                 GM_ADDR workspace, GM_ADDR tiling)
{
    // ---- 1. 解析 host 下发的 tiling 参数 ----
    // tiling 是一段 device 可见的 GM (全局内存), host 端写入 M/K/N 三个 uint32。
    // 注意: tiling 入参类型是 __gm__ uint8_t*, 必须用 __gm__ 修饰符 cast 到
    // __gm__ uint32_t* —— 否则 bisheng 会拒绝跨地址空间的 reinterpret_cast
    // (GM 地址空间 → 私有地址空间不合法)。
    __gm__ uint32_t* t = reinterpret_cast<__gm__ uint32_t*>(tiling);
    uint32_t M = t[0];  // A 的行数 / C 的行数
    uint32_t K = t[1];  // A 的列数 / B 的行数
    uint32_t N = t[2];  // B 的列数 / C 的列数

    // ---- 2. 构造 GlobalTensor 视图: 把裸指针 + 长度包装成可索引的张量 ----
    // GlobalTensor 是 GM (HBM) 上的逻辑视图, 不搬数据, 只记录基址与长度。
    GlobalTensor<half> A_global;
    GlobalTensor<half> B_global;
    GlobalTensor<half> C_global;

    // __gm__ half*: __gm__ 修饰符表示该指针位于全局内存地址空间
    A_global.SetGlobalBuffer((__gm__ half*)a, M * K);
    B_global.SetGlobalBuffer((__gm__ half*)b, K * N);
    C_global.SetGlobalBuffer((__gm__ half*)c, M * N);

    // ---- 3. 朴素三重循环 GEMM ----
    // 教学版: 直接逐元素从 GM 读写, 不分块、不调用 Cube、不使用片上 UB/L1。
    // 性能极差 (访存带宽完全没复用), 仅用于跑通正确性。
    //
    // 优化方向 (后续版本):
    //   * Tiling: 把 A/B 的子块从 GM 搬到片上 UB (Unified Buffer) / L1,
    //     复用 K 维数据, 减少 GM 访问次数。
    //   * 调用 Cube 单元: 用 LocalTensor + MatMul 接口做 16x16 矩阵乘,
    //     替代标量乘加, 充分利用 Cube 算力。
    //   * 多核并行: 用 GetBlockNum()/GetBlockIdx() 把 M 维切给多个 AI Core。
    //   * 双缓冲/流水线: 数据搬运与计算重叠, 掩盖访存延迟。
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;  // fp32 累加器, 避免 fp16 累加溢出
            for (uint32_t k = 0; k < K; ++k) {
                // half 提升到 float 再乘加 (fp16*fp16 在 C++ 里本身会提升, 显式写更清晰)
                acc += float(A_global.GetValue(m * K + k))
                     * float(B_global.GetValue(k * N + n));
            }
            // 写回 fp16 (fp32 累加结果截断到 fp16)
            C_global.SetValue(m * N + n, half(acc));
        }
    }
}
