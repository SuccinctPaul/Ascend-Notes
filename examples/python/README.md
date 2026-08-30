# python/ — GEMM 正确性基准 (Ground Truth)

本目录是整个 Ascend-Notes 项目的**正确性基准**:用纯 NumPy 在 CPU 上实现 GEMM,
所有其他 DSL(`../ascend_c/`、`../triton_ascend/`、`../tilelang_ascend/`)的 kernel 输出
都会与这里的 `gemm_reference` 对齐校验。

> 它**不是 NPU kernel**,跑在 CPU 上,目的是提供一个可信、可复现的参照答案。

## GEMM 数学定义

通用矩阵乘 (General Matrix Multiply):

```
C = alpha * A @ B + beta * C
```

其中 `A ∈ R^{M×K}`,`B ∈ R^{K×N}`,`C ∈ R^{M×N}`。本目录取最简形式 `alpha=1, beta=0`,
即 `C = A @ B`。朴素三重循环复杂度 `O(M·N·K)`。

## 数据精度约定

- 输入/输出 dtype:**float16**(与 NPU 其余 DSL 一致,fp16 是 Cube 单元原生精度)。
- 累加器:**float32**。`gemm_reference` 先把输入升到 fp32 再做乘加,避免 fp16 累加溢出,
  这与 NPU 上 "fp16 输入 + fp32 累加器" 的标准做法一致。
- 校验容差:`atol=1e-2, rtol=1e-2`(fp16 经验值)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `src/gemm.py` | `gemm_native`(朴素三重循环,教学)+ `gemm_reference`(NumPy BLAS 基准)+ `verify`(allclose 校验) |
| `src/tools.py` | 彩色耗时打印工具 |

## 如何运行

```bash
cd python
uv sync                 # 安装依赖 (numpy, termcolor)
uv run python src/gemm.py
```

预期输出包含:
```
[INFO] 校验结果: PASS (max_abs_error=..., atol=1e-02, rtol=1e-02)
[INFO] Python GEMM 基准完成, 全部 PASS
```

## 与其他 DSL 的关系

| DSL | 参考基准来源 | 说明 |
|---|---|---|
| python (本目录) | `gemm_reference` (NumPy BLAS) | ground truth |
| ascend_c | 与本目录 `gemm_reference` 对齐 | CANN C++ kernel |
| triton_ascend | 与 `torch.matmul` 或本目录基准对齐 | Triton on Ascend |
| tilelang_ascend | 与本目录 `gemm_reference` 对齐 | TileLang on Ascend |

> 注:朴素三重循环 `gemm_native` 在 fp16 下会有累加精度损失,所以它本身也用 `gemm_reference` 来校验;
> NPU kernel 同理 —— 用 fp32 累加器保证精度,再用容差比较。
