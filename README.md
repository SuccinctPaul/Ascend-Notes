# Ascend-Notes

在 Ascend NPU 上,以 **GEMM (C = A×B)** 为案例,用 4 种不同 DSL 实现 kernel 并对比学习。
每种 DSL 单独成目录,带详细中文注释与 README,讲述该 DSL 的工具链与运行方式。

## 四种 DSL 目录

| 目录 | DSL | 语言 | 抽象层级 | 工具链 | 说明 |
|---|---|---|---|---|---|
| [`python/`](python/) | NumPy | Python | 最高 (无 NPU) | numpy + uv | **正确性基准 (ground truth)**, CPU 参考实现 |
| [`ascend_c/`](ascend_c/) | Ascend C | C++ | 最低 | CANN `bisheng` 编译器 + ACL | CANN 原生 kernel, 直接操作硬件资源 (device 构建待调通, 见其 README) |
| [`triton_ascend/`](triton_ascend/) | Triton | Python (`@triton.jit`) | 中 (块级) | triton-ascend 后端 + torch_npu | OpenAI Triton 的昇腾后端, `tl.dot`→Cube |
| [`tilelang_ascend/`](tilelang_ascend/) | TileLang | Python (`@tilelang.jit`) | 中 (偏调度) | tilelang + tilelang-ascend 后端 | 北大开源, 显式 tiling/搬运/流水线 |

## 统一约定

- **GEMM**:`C = A @ B`,`A∈R^{M×K}`,`B∈R^{K×N}`,`C∈R^{M×N}`(测试规模 M=N=K=128)。
- **数据精度**:输入/输出 **float16**(Cube 单元原生精度);累加器 **float32**(混合精度,避免溢出)。
- **实现层级**:朴素版为主 + 注释/README 讲解优化方向(tiling/Cube/UB/流水线)。
- **正确性校验**:每个 DSL 的 kernel 输出都与参考基准对齐,`allclose(atol=1e-2, rtol=1e-2)`,打印 PASS/FAIL。
- **包管理**:每个 Python DSL 目录用 [uv](https://github.com/astral-sh/uv) 独立 venv(`pyproject.toml`)。

## 快速对比:四种 DSL 写同一个 GEMM

```
python/          →  三重循环 (np.matmul 基准)
ascend_c/        →  GlobalTensor + 标量乘加 (逐元素读 GM, 最朴素)
triton_ascend/   →  make_block_ptr + tl.dot (分块 + Cube 自动调用)
tilelang_ascend/ →  alloc_shared + T.copy + T.gemm + T.Pipelined (显式 tiling/搬运/流水线)
```

## 运行环境

远程服务器:`ssh vllm-hust-cyj-21rc-cloud-piou`,开发路径 `/root/Ascend-Notes`。
- 架构:aarch64 (Ubuntu)
- CANN:9.0.0
- NPU:Ascend910B 系列
- 所有 NPU kernel 在此服务器上构建与测试;`python/` 基准可在任意机器跑。

### 运行前置(每次 shell 都要先 source)
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

## 如何逐个运行

```bash
# 1. python 基准 (CPU, 任意机器)
cd python && uv sync && uv run python src/gemm.py

# 2. ascend_c (需 CANN + NPU)
cd ascend_c && cmake -S . -B build && cmake --build build && ./build/ascend_gemm

# 3. triton_ascend (需 torch_npu + triton-ascend, 详见其 README)
cd triton_ascend && uv run python src/test_gemm.py

# 4. tilelang_ascend (需 tilelang + tilelang-ascend, 详见其 README)
cd tilelang_ascend && uv run python src/test_gemm.py
```

每个 DSL 的预期结果均包含 `PASS`。详见各目录 README。
