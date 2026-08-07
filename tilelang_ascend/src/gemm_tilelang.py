"""
GEMM kernel —— TileLang + tilelang-ascend 后端。

TileLang 是北大开源的分块 (tiled) kernel DSL, 基于 TVM。
tilelang-ascend 是它对接华为 Ascend NPU 的后端, 把 TileLang IR 编译成 AscendNPU IR,
再经 CANN 工具链生成可执行码。

本实现基于 tilelang 官方 GEMM 示例, 适配 ascend_npu target:
  - 显式分块: 把输出切成 block_M×block_N, 每个 Kernel block 算一块
  - 片上缓冲: A/B 子块搬到 shared memory (对应 Ascend L1/UB), 复用 K 维数据
  - T.gemm: 调用底层矩阵乘原语 (映射到 Cube 单元)
  - T.Pipelined: 多级流水线 (num_stages=3), 搬运与计算重叠, 掩盖访存延迟
  - 累加器 fp32, 输入/输出 fp16 (混合精度)

相比 triton 的隐式优化, TileLang 把 tiling / 搬运 / 流水线都显式写在代码里,
更贴近硬件调度细节, 控制力更强。
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def gemm_matmul(A, B,
                block_M=128, block_N=128, block_K=32,
                dtype=T.float16, accum_dtype=T.float32):
    """
    TileLang GEMM kernel 定义。

    Args (编译期参数):
        A, B: 占位张量 (形状由 compile 时的 M/N/K 决定)
        block_M/N/K: 分块大小, block_K 是 K 维搬运粒度
        dtype: 输入精度 (fp16)
        accum_dtype: 累加器精度 (fp32, 避免 fp16 累加溢出)

    返回:
        C: 输出张量, 形状 (M, N), dtype=fp16
    """
    # M, N, K 作为编译期常量 (compile() 时传入具体值)
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)   # 输入 A: (M, K)
    B: T.Tensor((K, N), dtype)   # 输入 B: (K, N)
    C = T.empty((M, N), dtype)    # 输出 C: (M, N)

    # T.Kernel: 声明并行维度, 类似 CUDA 的 grid
    # 两个 block 维度: bx 遍历 N 维, by 遍历 M 维 (先 N 后 M)
    # threads=128: 每个 block 的线程数 (映射到 AICore 的并行度)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # ---- 片上缓冲分配 (对应 Ascend L1 / UB) ----
        # alloc_shared: shared memory, 跨核可见的高速片上缓存
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        # alloc_fragment: 寄存器级 fragment, 存累加器 (fp32)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)  # 累加器清零

        # ---- K 维流水线循环 ----
        # T.Pipelined: 多级流水线, num_stages=3 表示 3 级重叠
        #   (搬运下一块 / 计算 / 写回 同时进行), 掩盖访存延迟
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # T.copy: GM -> shared, 块搬运 (高效 DMA)
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            # T.gemm: 块矩阵乘, A_shared @ B_shared 累加到 C_local
            # 在 ascend 后端映射到 Cube 单元 (16x16 粒度)
            T.gemm(A_shared, B_shared, C_local)

        # ---- 结果写回 GM ----
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def gemm(a, b, block_M=128, block_N=128, block_K=32):
    """
    便捷封装: 输入 torch 张量, 返回 C = A @ B。

    首次调用会触发 tilelang-ascend 编译 (生成 ascend kernel), 较慢;
    后续调用走缓存。

    Args:
        a: (M, K) fp16 tensor (npu 设备)
        b: (K, N) fp16 tensor (npu 设备)
        block_M/N/K: 分块大小 (16 的倍数, 对齐 Cube 粒度)

    Returns:
        c: (M, N) fp16 tensor
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"A 的列数 {K} 必须等于 B 的行数 {K2}"

    # compile: 把 kernel 定义实例化为可执行 kernel (M/N/K 编译期绑定)
    kernel = gemm_matmul.compile(
        M=M, N=N, K=K,
        block_M=block_M, block_N=block_N, block_K=block_K,
    )
    c = kernel(a, b)
    return c
