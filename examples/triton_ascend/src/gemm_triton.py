"""
GEMM kernel —— Triton on Ascend (triton-ascend 后端) 朴素分块实现。

Triton 是 OpenAI 的 GPU kernel DSL, triton-ascend 是它在昇腾 NPU 上的后端实现:
同一份 Python kernel 代码, 经 triton-ascend 编译后跑在 Ascend 的 Cube/Vector 单元上。

本实现采用 "一个 program 算一个 BLOCK_M×BLOCK_N 输出块" 的朴素分块策略:
  - grid 大小 = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
  - 每个 program 沿 K 维循环累加, 用 tl.dot 做块矩阵乘 (自动映射到 Cube 单元)
  - 累加器用 fp32, 输入/输出 fp16 (混合精度)

相比 ascend_c 的朴素三重循环, 这里天然带分块 + Cube 调用, 是 "半优化" 起步点。
"""

import torch

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A 的行/列步长 (元素数)
    stride_bk, stride_bn,   # B 的行/列步长
    stride_cm, stride_cn,   # C 的行/列步长
    BLOCK_M: tl.constexpr,  # 输出块行数 (编译期常量)
    BLOCK_N: tl.constexpr,  # 输出块列数
    BLOCK_K: tl.constexpr,  # K 维分块大小
):
    # ---- 1. 程序块定位 ----
    # program_id 映射到 NPU 的 AI Core: 每个 program 负责一个输出块
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # 行优先的块排列: 先遍历 N 维, 再 M 维
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ---- 2. 构造分块指针 (block pointer) ----
    # tl.make_block_ptr: 把基地址 + 形状 + 步长 + 偏移 + 块形状包装成分块视图
    # order=(1, 0): 行主序内存布局 (昇腾上更快), 即最内维 (最后一维) 连续
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # ---- 3. K 维循环累加 ----
    # accumulator 用 fp32: 避免 fp16 在大 K 下累加溢出 (混合精度标准做法)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # tl.load + boundary_check: 处理 M/N/K 不能被 BLOCK 整除的边界
        a = tl.load(a_block_ptr, boundary_check=(0, 1))   # [BLOCK_M, BLOCK_K] fp16
        b = tl.load(b_block_ptr, boundary_check=(0, 1))   # [BLOCK_K, BLOCK_N] fp16
        # tl.dot: 块矩阵乘, triton-ascend 自动映射到 Cube 单元 (16x16 粒度)
        #   => 输入 fp16, 累加到 fp32 accumulator
        accumulator += tl.dot(a, b)
        # 沿 K 维推进分块指针
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ---- 4. 写回 (fp32 累加结果转 fp16 存储) ----
    tl.store(c_block_ptr, accumulator.to(tl.float16))


def gemm(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):
    """
    便捷封装: 输入 torch 张量 (npu 设备), 返回 C = A @ B。

    Args:
        a: (M, K) fp16, 在 npu 上
        b: (K, N) fp16, 在 npu 上
        BLOCK_M/N/K: 分块大小, 必须是 16 的倍数 (Cube 16x16 粒度约束)

    Returns:
        c: (M, N) fp16, 在 npu 上
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"A 的列数 {K} 必须等于 B 的行数 {K2}"
    assert a.is_npu and b.is_npu, "输入张量必须在 npu 设备上"
    assert a.dtype == b.dtype == a.dtype, "A/B dtype 需一致"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # grid: 每个 program 算一个 BLOCK_M×BLOCK_N 输出块
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
