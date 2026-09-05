# python/ — 多算子正确性基准 (Ground Truth)

本目录是整个 Ascend-Notes 项目的**正确性基准**:用纯 NumPy 在 CPU 上实现各算子,
所有其他 DSL(`../ascend_c/`、`../triton_ascend/`、`../tilelang_ascend/`)的 kernel 输出
都会与这里的 `*_reference` 对齐校验。

> 它**不是 NPU kernel**,跑在 CPU 上,目的是提供一个可信、可复现的参照答案。

## 覆盖的算子

| 算子 | reference 入口 | 数学定义 |
|---|---|---|
| GEMM | `gemm.gemm_reference` | `C = A @ B`(朴素三重循环 + BLAS 版) |
| Softmax | `softmax.softmax_reference` | 数值稳定版:减 max → exp → sum → div |
| GELU | `gelu.gelu_reference` | tanh 近似 (0.5x(1+tanh(√(2/π)(x+0.044715x³)))) |
| RMSNorm | `rmsnorm.rmsnorm_reference` | `y = x / sqrt(mean(x²)+eps) · gamma` |
| RoPE | `rope.rope_reference` | 交错配对二维旋转 `(x[2a],x[2a+1]) · (cos+i·sin)` |
| INT8 量化 | `quant.quant_int8_reference` | 逐行 absmax 对称量化 + 反量化往返 |
| GQA 解码 | `gqa.gqa_decode_reference` | 解码一步: q·KV cache 打分 + softmax + 加权 (分组) |
| FlashAttention | `flash.attention_reference` | 标准注意力 (与 flash 数学等价) + online 增量版 |

## 数据精度约定

- 输入/输出 dtype:**float16**(与 NPU 其余 DSL 一致,fp16 是 Cube/Vector 原生精度)。
- 归约/中间量:**float32**。所有 reference 先把输入升到 fp32 再做归约/乘加,
  这与 NPU 上 "fp16 输入 + fp32 累加器" 的标准做法一致("存窄算宽")。
- 校验容差:`atol=1e-2, rtol=1e-2`(fp16 经验值)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `src/gemm.py` | `gemm_native`(朴素三重循环,教学)+ `gemm_reference`(NumPy BLAS 基准)+ `verify` |
| `src/softmax.py` | `softmax_reference`(稳定版)+ `softmax_naive`(会溢出的反面教材) |
| `src/gelu.py` | GELU 的 tanh 近似 / 精确 erf 版参考实现 |
| `src/rmsnorm.py` | `rmsnorm_reference`(fp32 归约)+ `rmsnorm_naive`(fp16 归约误差对照) |
| `src/rope.py` | `rope_reference` 查表版 + `apply_rope_numpy` 现算版 + θ/cos/sin 表预计算 |
| `src/quant.py` | INT8 对称量化/反量化 + 往返误差上界 |
| `src/gqa.py` | GQA/MQA/MHA 解码注意力 (KV Cache) |
| `src/flash.py` | 标准注意力参考 + flash online 增量参考 (等价性验证) |
| `src/test_*.py` | pytest 正确性测试(性质校验 + torch 交叉验证);`__main__` 可无 pytest 直接跑 smoke |
| `src/tools.py` | 彩色耗时打印工具 |

## 如何运行

```bash
cd python
uv sync                 # 安装依赖 (numpy, termcolor, pytest)
uv run python src/gemm.py src/softmax.py src/gelu.py src/rmsnorm.py src/rope.py src/quant.py src/gqa.py src/flash.py
uv run pytest src/test_softmax.py src/test_gelu.py src/test_rmsnorm.py \
    src/test_rope.py src/test_quant.py src/test_gqa.py src/test_flash.py -v  # 完整 pytest
```

## 与其他 DSL 的关系

| DSL | 参考基准来源 | 说明 |
|---|---|---|
| python (本目录) | `*_reference` (NumPy) | ground truth |
| ascend_c | 与本目录 reference 同公式 | CANN C++ kernel |
| triton_ascend | 与本目录 reference / torch 对齐 | Triton on Ascend |
| tilelang_ascend | 与本目录 reference 同公式 | TileLang on Ascend |

> 注:NPU kernel 同样用 fp32 累加器保证精度,再用容差比较;`*_naive` 版本
> (softmax 不减 max、rmsnorm 用 fp16 归约) 是展示"为什么必须这么做"的反面教材。
