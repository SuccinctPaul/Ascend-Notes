"""
TileLang-Ascend GEMM 正确性测试。

流程:
  1. 生成 fp16 随机矩阵 A, B (torch tensor, npu 设备)
  2. 调 tilelang kernel 算 C = A @ B
  3. 用 numpy 算参考 (fp16 输入升 fp32 累加)
  4. allclose 校验 (fp16 容差), 打印 PASS/FAIL + 耗时

注意: tilelang kernel 接受 torch tensor; ascend 后端要求张量在 npu 上。
若环境未装 torch_npu, 可把 device 改为 cpu 做纯编译验证 (但不跑在 NPU 上)。

环境变量:
  ACL_OP_INIT_MODE=1 必须在 import torch_npu 前设置 —— tilelang 自带的 TVM 与
  CANN 的 te 模块共享 TVM FFI 全局注册表, 会互相覆盖。设此变量跳过 torch_npu
  的 TBE/GE 算子编译器初始化 (本测试只做张量分配 + tilelang 自管 kernel launch,
  不走 torch_npu 图编译), 避免冲突。
"""

import os
# 必须在 import torch_npu 之前设置 (见文件 docstring 说明)
os.environ.setdefault("ACL_OP_INIT_MODE", "1")

import time
import logging

import numpy as np

import tilelang  # noqa: F401  导入触发后端注册

# tilelang kernel 接受 torch tensor, 这里用 torch 构造输入
import torch
try:
    import torch_npu  # noqa: F401  注册 npu 设备
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

from gemm_tilelang import gemm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    M = N = K = 128
    logging.info("=== TileLang-Ascend GEMM 测试 (dtype=float16) ===")

    np.random.seed(0)
    # numpy 生成参考数据
    a_np = np.random.randn(M, K).astype(np.float16)
    b_np = np.random.randn(K, N).astype(np.float16)

    # 转 torch tensor, 放到 npu (若可用)
    device = "npu" if _HAS_NPU else "cpu"
    a = torch.from_numpy(a_np).to(device)
    b = torch.from_numpy(b_np).to(device)

    # 参考基准: numpy fp32 累加
    c_ref = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)

    # 预热 (首次触发 tilelang-ascend 编译)
    logging.info("预热编译 (首次调用触发 tilelang-ascend 编译)...")
    _ = gemm(a, b)

    # 正式计时 (K_L1=64: K=128 分 2 次搬到 L1, 展示累加语义)
    start = time.perf_counter()
    c = gemm(a, b, block_M=128, block_N=128, K_L1=64)
    elapsed = time.perf_counter() - start
    logging.info("TileLang kernel 耗时: %.4f ms", elapsed * 1000)

    # 取回 host 做校验
    c_np = c.cpu().numpy()

    # 校验 (fp16 容差)
    max_err = float(np.max(np.abs(c_np.astype(np.float32) - c_ref.astype(np.float32))))
    ok = bool(np.allclose(c_np, c_ref, atol=1e-2, rtol=1e-2))
    status = "PASS" if ok else "FAIL"
    logging.info("校验结果: %s (max_abs_error=%.6e, atol=1e-2, rtol=1e-2)",
                 status, max_err)
    logging.info("A: %s, B: %s, C: %s (device=%s)",
                 a.shape, b.shape, c.shape, c.device)

    if not ok:
        raise SystemExit(f"TileLang-Ascend GEMM 校验失败: max_abs_error={max_err}")
    logging.info("TileLang-Ascend GEMM 测试完成, 全部 PASS")


if __name__ == "__main__":
    main()
