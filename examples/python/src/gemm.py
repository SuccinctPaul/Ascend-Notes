"""
GEMM (General Matrix Multiply) —— 纯 Python/NumPy 参考实现。

本模块是整个 Ascend-Notes 项目的 **正确性基准 (ground truth)**:
它跑在 CPU 上,不涉及 NPU kernel。所有其他 DSL (ascend_c / triton_ascend /
tilelang_ascend) 的 GEMM 输出都会与这里的 `gemm_reference` 对齐。

GEMM 数学定义:
    C = alpha * A @ B + beta * C
本实现取最简形式 alpha=1, beta=0, 即 C = A @ B,其中 A∈R^{M×K}, B∈R^{K×N},
C∈R^{M×N}。

朴素三重循环复杂度 O(M·N·K),仅用于教学演示算法本身,不追求性能。
"""

import time
import logging

import numpy as np

import tools  # 本地工具: 彩色耗时打印

# 全局日志配置: INFO 打印进度, DEBUG 打印矩阵尺寸/耗时细节
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def gemm_native(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    朴素三重循环 GEMM, 仅用于教学演示。

    理论:
        GEMM = MatMul + bias + scale + transpose + layout + dtype + epilogue
        例如: C = alpha * A @ B + beta * C
        这里取最简形式: C = A @ B

    本实现直接用 Python for 循环逐元素累加, 与硬件 kernel 的朴素写法一一对应,
    方便理解后续 ascend_c/triton/tilelang 的朴素版本在做什么。

    Args:
        A: 输入矩阵 A, 形状 (M, K)
        B: 输入矩阵 B, 形状 (K, N)

    Returns:
        C: 输出矩阵 C = A @ B, 形状 (M, N), dtype 与 A 一致

    Raises:
        ValueError: 形状不匹配或输入不是 ndarray
    """
    # --- 输入校验 (系统边界处做校验, 内部循环信任已校验的输入) ---
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("A and B must be numpy arrays")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"A 的列数 {A.shape[1]} 必须等于 B 的行数 {B.shape[0]}"
        )

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "K 维度不一致, 前置校验已应保证不会到这里"

    logging.debug("Matrix A Size: %d x %d", M, K)
    logging.debug("Matrix B Size: %d x %d", K2, N)

    # 输出与 A 同 dtype: 若 A 是 float16, C 也是 float16
    # 注意: float16 下朴素累加会有精度损失, 这正是 NPU 用 fp32 累加器的原因
    C = np.zeros((M, N), dtype=A.dtype)
    logging.debug("Matrix C Size: %d x %d (dtype=%s)", M, N, C.dtype)

    start = time.perf_counter()

    # 朴素三重循环: i 遍历 M 行, j 遍历 N 列, k 在 K 维上做点积
    # 关键: 即便输入是 fp16, 也把每个元素升到 fp32 再乘加 —— 这正是 NPU 上
    # "fp16 输入 + fp32 累加器" 的标准做法 (混合精度), 否则 fp16 逐次乘法会
    # 在 K 较大时累积出显著误差。
    acc_dtype = np.float32
    for i in range(M):
        for j in range(N):
            s = acc_dtype(0.0)  # fp32 累加器
            for k in range(K):
                s += acc_dtype(A[i, k]) * acc_dtype(B[k, j])
            C[i, j] = s  # 回写到目标 dtype (可能 fp16, 有截断)

    end = time.perf_counter()
    logging.debug("朴素 GEMM 耗时: %s", tools.format_time_color(end - start))

    return C


def gemm_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    参考基准实现: 直接调用 NumPy 的 @ 运算 (底层是 BLAS, 高度优化)。

    这是所有 DSL kernel 的 **对齐基准**:
    - ascend_c / triton / tilelang 的输出都要和它做 allclose 比较
    - 为了避免 fp16 累加误差, 这里先升到 fp32 做乘加, 再转回输入 dtype
      (与 NPU 上 "fp16 输入 + fp32 累加器" 的标准做法一致)

    Args:
        A: 输入矩阵 A, 形状 (M, K)
        B: 输入矩阵 B, 形状 (K, N)

    Returns:
        C_ref: 参考输出 C = A @ B, 形状 (M, N), dtype 与 A 一致
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("A 的列数必须等于 B 的行数")

    in_dtype = A.dtype
    # 升到 float32 计算, 保证累加精度, 再截回输入精度
    # 这样无论输入是 fp16/fp32, 参考结果都数值稳定
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(in_dtype)
    return C_ref


def verify(C: np.ndarray, C_ref: np.ndarray,
           atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """
    校验 kernel 输出与参考基准是否一致。

    fp16 容差取 atol=1e-2, rtol=1e-2 (经验值, fp16 尾数约 3 位十进制有效数字)。

    Args:
        C:     被 校验的输出 (来自 kernel 或朴素实现)
        C_ref: 参考基准输出
        atol:  绝对误差容差
        rtol:  相对误差容差

    Returns:
        True 表示通过 (PASS), False 表示失败 (FAIL)
    """
    assert C.shape == C_ref.shape, f"形状不一致: {C.shape} vs {C_ref.shape}"
    max_abs_err = float(np.max(np.abs(C.astype(np.float32) - C_ref.astype(np.float32))))
    ok = bool(np.allclose(C, C_ref, atol=atol, rtol=rtol))
    status = "PASS" if ok else "FAIL"
    logging.info("校验结果: %s (max_abs_error=%.6e, atol=%.0e, rtol=%.0e)",
                 status, max_abs_err, atol, rtol)
    return ok


if __name__ == "__main__":
    # 固定随机种子, 保证可复现
    np.random.seed(0)

    # 测试矩阵规模: M=N=K=128, fp16 (与 NPU 其余 DSL 保持一致)
    M, K, N = 128, 128, 128

    logging.info("=== Python GEMM 基准 (dtype=float16) ===")
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)

    # 1) 参考基准 (NumPy BLAS)
    C_ref = gemm_reference(A, B)

    # 2) 朴素三重循环 (教学)
    C_native = gemm_native(A, B)

    # 3) 校验: 朴素版 vs 参考版
    ok = verify(C_native, C_ref)

    logging.info("A: %s, B: %s, C_ref: %s, C_native: %s",
                 A.shape, B.shape, C_ref.shape, C_native.shape)

    if not ok:
        raise SystemExit("朴素 GEMM 与参考基准不一致, FAIL")
    logging.info("Python GEMM 基准完成, 全部 PASS")
