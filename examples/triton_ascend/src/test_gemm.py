"""
Triton-Ascend GEMM 正确性测试。

流程:
  1. 在 npu 上生成 fp16 随机矩阵 A, B
  2. 调 triton kernel 算 C = A @ B
  3. 用 torch.matmul (底层走 NPU Cube) 作为参考
  4. allclose 校验 (fp16 容差), 打印 PASS/FAIL + 耗时
"""

import time
import logging

import torch
import torch_npu  # 注册 'npu' 设备, 必须在 torch 之后 import

from gemm_triton import gemm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    M = N = K = 128
    logging.info("=== Triton-Ascend GEMM 测试 (dtype=float16) ===")

    # 在 npu 上生成 fp16 数据
    a = torch.randn((M, K), device="npu", dtype=torch.float16)
    b = torch.randn((K, N), device="npu", dtype=torch.float16)

    # 预热 (首次调用会触发 triton 编译, 慢; 正式计时排除编译开销)
    logging.info("预热编译 (首次调用触发 triton-ascend 编译)...")
    _ = gemm(a, b)
    torch.npu.synchronize()

    # 正式计时
    start = time.perf_counter()
    c = gemm(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    torch.npu.synchronize()  # NPU 异步, 需同步后计时才准
    elapsed = time.perf_counter() - start
    logging.info("Triton kernel 耗时: %.4f ms", elapsed * 1000)

    # 参考基准: torch.matmul (走 NPU Cube 单元)
    c_ref = torch.matmul(a, b)

    # 校验 (fp16 容差)
    max_err = (c.float() - c_ref.float()).abs().max().item()
    ok = torch.allclose(c, c_ref, atol=1e-2, rtol=1e-2)
    status = "PASS" if ok else "FAIL"
    logging.info("校验结果: %s (max_abs_error=%.6e, atol=1e-2, rtol=1e-2)",
                 status, max_err)
    logging.info("A: %s, B: %s, C: %s (device=%s)",
                 tuple(a.shape), tuple(b.shape), tuple(c.shape), c.device)

    if not ok:
        raise SystemExit(f"Triton-Ascend GEMM 校验失败: max_abs_error={max_err}")
    logging.info("Triton-Ascend GEMM 测试完成, 全部 PASS")


if __name__ == "__main__":
    main()
