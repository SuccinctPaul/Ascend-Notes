"""
GEMM kernel —— TileLang + tilelang-ascend 后端。

TileLang 是北大杨智团队开源的分块 (tiled) kernel DSL, 基于 TVM。
tilelang-ascend 是它对接华为 Ascend NPU 的后端, 把 TileLang IR 编译成
AscendNPU IR / Ascend C, 再经 CANN 工具链生成可执行码。

本实现参考 tilelang-ascend 官方示例 examples/gemm/example_gemm.py,
适配 128×128 fp16 场景, 并补全中文注释讲解 Ascend 内存层次与 cube 调度:

  - is_npu=True        : 声明 kernel 跑在 NPU (而非 GPU 的 thread block)
  - T.alloc_L1         : L1 buffer (Cube 核片上缓存, 类比 GPU shared memory)
  - T.alloc_L0C        : L0C buffer (Cube 核累加器寄存器, 类比 GPU fragment)
  - T.Scope("C")       : Cube 核执行域 (NPU 上 Cube/Vector 分核, 这里走 Cube)
  - T.gemm_v0(A,B,C,init): Cube 单元矩阵乘; init=True 清零累加, False 累加
  - T.barrier_all()    : 片内同步 (MTE3/MTE2 等队列间同步)
  - T.copy(GM, L1)     : GM -> L1 块搬运 (高效 DMA)
  - 累加器 fp32, 输入/输出 fp16 (混合精度, Cube 原生)

相比 triton 的隐式优化, TileLang 把 tiling / 搬运 / 核内调度都显式写在代码里,
更贴近 Ascend 硬件调度细节, 控制力更强。
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def gemm_matmul(M, N, K, block_M, block_N, K_L1,
                dtype="float16", accum_dtype="float"):
    """
    TileLang-Ascend GEMM kernel 定义。

    Args (编译期参数):
        M, N, K      : 矩阵形状 (编译期常量)
        block_M      : M 维分块大小 (Cube 粒度, 16 的倍数)
        block_N      : N 维分块大小
        K_L1         : K 维 L1 搬运粒度 (一次搬多少 K 到 L1)
        dtype        : 输入精度 (fp16, Cube 原生)
        accum_dtype  : 累加器精度 (fp32, 避免 fp16 累加溢出)

    out_idx=[-1] : 返回最后一个张量 (C) 作为输出
    """
    m_num = M // block_M    # M 维 block 数
    n_num = N // block_N    # N 维 block 数

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),    # 输入 A: (M, K) fp16
        B: T.Tensor((K, N), dtype),    # 输入 B: (K, N) fp16
        C: T.Tensor((M, N), dtype),    # 输出 C: (M, N) fp16
    ):
        # T.Kernel: 声明并行维度。
        # is_npu=True : 关键! 告诉 tilelang 这是 NPU kernel (走 ascend 后端),
        #               而非 GPU 的 CUDA thread block。
        # 这里把 m_num*n_num 个输出 block 拍平成 1 维, 由 cid 自行拆回 (bx, by)。
        # 单 block (m_num=n_num=1) 时 grid=1, 即单核; 多 block 时多核并行。
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num   # M 维 block 索引
            by = cid % n_num    # N 维 block 索引

            # ---- 片上缓冲分配 (对应 Ascend Cube 核内存层次) ----
            # L1  : Cube 核片上高速缓存 (类比 GPU shared memory), 存 A/B 子块
            # L0C : Cube 核累加器寄存器 (类比 GPU fragment), 存 fp32 中间结果
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # T.Scope("C") : Cube 核执行域
            # NPU 的 AI Core 分 Cube 核 (矩阵乘) 与 Vector 核 (向量运算),
            # 本 GEMM 只用 Cube, 故整体包在 Scope("C") 里。
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)   # K 维分块数
                for k in T.serial(loop_k):
                    # GM -> L1 块搬运 (DMA, 高效)
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    # 片内同步: 确保搬运完成再计算 (MTE2→MTE1 队列)
                    T.barrier_all()

                    # T.gemm_v0 : Cube 单元矩阵乘, A_L1 @ B_L1 累加到 C_L0
                    #   init=(k == 0) : 第一块时清零累加器, 后续块累加
                    #   (这是 Ascend Cube 的标准累加语义, 避免显式 clear)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                # L0C -> GM 写回 (fp32 累加结果转 fp16)
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def gemm(a, b, block_M=128, block_N=128, K_L1=64):
    """
    便捷封装: 输入 torch 张量, 返回 C = A @ B。

    首次调用会触发 tilelang-ascend 编译 (生成 ascend kernel), 较慢;
    后续调用走缓存。

    Args:
        a: (M, K) fp16 tensor (npu 设备)
        b: (K, N) fp16 tensor (npu 设备)
        block_M : M 维分块大小 (16 的倍数, 对齐 Cube 粒度)
        block_N : N 维分块大小
        K_L1    : K 维 L1 搬运粒度 (一次搬多少 K 到片上)

    Returns:
        c: (M, N) fp16 tensor
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"A 的列数 {K} 必须等于 B 的行数 {K2}"
    assert M % block_M == 0, f"M={M} 必须是 block_M={block_M} 的倍数"
    assert N % block_N == 0, f"N={N} 必须是 block_N={block_N} 的倍数"

    # 编译并调用 (M/N/K 编译期绑定)
    kernel = gemm_matmul(M, N, K, block_M, block_N, K_L1)
    c = kernel(a, b)
    return c
