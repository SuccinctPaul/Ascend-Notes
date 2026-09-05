# triton_ascend/ — Triton on Ascend NPU

本目录用 **Triton**(OpenAI 的 GPU kernel DSL)+ **triton-ascend**(昇腾后端)实现 GEMM。
同一份 Python kernel,经 triton-ascend 编译后跑在 Ascend 的 Cube/Vector 单元上。

## Triton-Ascend 是什么

[Triton](https://github.com/triton-lang/triton) 是 OpenAI 开源的 GPU kernel 编程语言,
用 Pythonic 语法写高性能 kernel。**triton-ascend** 是 Triton 在华为昇腾 NPU 上的后端实现
(仓库 [gitcode.com/Ascend/triton-ascend](https://gitcode.com/Ascend/triton-ascend)),
它把 Triton IR 编译成 Ascend NPU IR,再经 CANN 工具链生成可执行码。

### 与 GPU Triton / ascend_c 的区别

| 维度 | GPU Triton | triton-ascend (本目录) | ascend_c |
|---|---|---|---|
| 语言 | Python (`@triton.jit`) | Python (`@triton.jit`) | C++ |
| 后端 | NVIDIA CUDA | Ascend Cube/Vector | Ascend 原生 |
| 抽象层级 | 高 (块级) | 高 (块级) | 低 (元素级) |
| Cube 映射 | N/A | `tl.dot` → Cube 16×16 | 需手动调接口 |

> 同一份 Triton kernel 代码,在 GPU 上走 CUDA 后端,在 NPU 上走 ascend 后端 —— 这是
> "一次编写,多处运行" 的卖点。

## 工具链安装

### 前置:CANN 环境
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 安装 torch + torch_npu + triton-ascend
```bash
cd triton_ascend
uv venv --python 3.11          # 创建独立 venv
uv sync                         # 安装 numpy, torch==2.8.0 (PyPI 可解析部分)

# torch_npu: 必须 pin 2.8.0rc1 与 torch 2.8.0 匹配!
# (2026-09 实测: 不 pin 可能被解析到不匹配版本, import 报 undefined symbol)
uv pip install "torch-npu==2.8.0rc1" --prerelease=allow
uv pip install triton-ascend    # triton-ascend 3.2.0; 失败则源码:
#   git clone https://gitcode.com/Ascend/triton-ascend.git
#   cd triton-ascend && pip install -e .

# torch_npu/triton-ascend 的运行时依赖 (裸环境必装, 2026-09 新容器实测):
uv pip install pyyaml decorator attrs psutil scipy pybind11
```

> **版本匹配很关键**(本机实测组合):
> - CANN 9.0.0 + torch 2.8.0 + torch_npu 2.8.0rc1 + triton-ascend 3.2.0
> - torch_npu 必须与 torch 版本严格一致 (2.8.0rc1 ↔ 2.8.0), 否则 import 报符号找不到。
> - triton-ascend 3.2.0 在 CANN 9.0.0 上有 enum 重命名 (见"常见问题")。

### 验证安装
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
uv run python -c "import torch, torch_npu, triton; print(torch.npu.is_available())"
```

## 编译通路

```
gemm_triton.py (@triton.jit)
   │  [Triton 前端] Python → Triton IR (MLIR)
   ▼
Triton IR
   │  [triton-ascend 后端] IR → Ascend NPU IR (lowering)
   ▼
Ascend NPU IR
   │  [CANN 工具链] 链接 → 可执行 kernel
   ▼
NPU Cube/Vector 单元执行
```

## kernel 设计要点(见 `src/gemm_triton.py` 注释)

- **朴素分块策略**:一个 `program`(`tl.program_id`)算一个 `BLOCK_M×BLOCK_N` 输出块,
  沿 K 维循环累加。`grid = (cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N),)`。
- **`tl.make_block_ptr`**:构造分块指针,`order=(1,0)` 行主序(昇腾高效布局)。
- **`tl.dot`**:块矩阵乘,triton-ascend **自动映射到 Cube 单元**(16×16 粒度)。
  输入 fp16 → 累加 fp32(混合精度)。
- **BLOCK 取 16 的倍数**:对齐 Cube 单元的 16×16 计算粒度。

### 优化方向(注释中展开)

- `@triton.autotune`:自动搜索最优 BLOCK_M/N/K。
- `additional_config`(昇腾专属):`num_cores`(芯组并行)、`cube_mode`。
- 多级流水线 / 双缓冲:`num_stages` 参数掩盖访存延迟。
- L1 缓存 128 字节对齐。

## 如何运行

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd triton_ascend
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_gemm.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_softmax.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_rmsnorm.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_rope.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_quant.py src/test_gqa.py src/test_flash.py
```

预期输出(每个测试文件最后):
```
All N cases PASSED
```

### RMSNorm / RoPE 性能基准

```bash
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/bench_rmsnorm_triton.py --json /tmp/rmsnorm.json
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/bench_rope_triton.py --json /tmp/rope.json
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/bench_quant_triton.py --json /tmp/quant.json
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/bench_gqa_triton.py --json /tmp/gqa.json
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/bench_flash_triton.py --json /tmp/flash.json
```

2026-09-05 实测(910B2, fp16, best-of-20):RMSNorm 16384×4096 = **1.26 ms(213 GB/s)**;
RoPE 16384×2048 = **11.75 ms**(注意:`rope_triton` 需用 `tables=` 预构建查表张量,
否则每次调用 host 现算三角函数,16384×2048 实测从 11.75 ms 恶化到 328 ms ——
这正是 docs/04 §5.2 "预计算 cos/sin、别在 kernel 里现算三角" 的量化证据)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `src/gemm_triton.py` | `@triton.jit` GEMM kernel + `gemm()` 便捷封装 |
| `src/softmax_triton.py` | 行 softmax:每 program 一行,3-pass(max/exp+sum/normalize),D>BLOCK 时 pad -inf |
| `src/gelu_triton.py` | GELU 逐元素(tanh 近似) |
| `src/rmsnorm_triton.py` | RMSNorm:每 program 一行,2-pass(fp32 Σx² → rsqrt → 乘 inv_rms·gamma) |
| `src/rope_triton.py` | RoPE:kernel 内半维拆分布局 + wrapper 做 interleaved 转换;`tables=` 预构建查表 |
| `src/quant_triton.py` | INT8 量化/反量化:逐行 fp32 absmax → round → int8 |
| `src/gqa_triton.py` | GQA 解码注意力:每 program 一头,flash-decode 在线 softmax 单趟扫 KV cache |
| `src/flash_triton.py` | FlashAttention FA2:tl.dot 走 Cube + online softmax,L×S 分数不落 GM |
| `src/test_gemm.py` / `test_softmax.py` / `test_gelu.py` / `test_rmsnorm.py` / `test_rope.py` / `test_quant.py` / `test_gqa.py` / `test_flash.py` | torch_npu 驱动 + numpy 参考 + allclose 校验(多 shape/dtype) |
| `src/bench_gelu_triton.py` / `bench_rmsnorm_triton.py` / `bench_rope_triton.py` / `bench_quant_triton.py` / `bench_gqa_triton.py` / `bench_flash_triton.py` | 微基准:多规模 best-of-N ms + GB/s + 正确性断言 |
| `pyproject.toml` | uv 配置 (numpy/torch); torch_npu/triton-ascend 手动装 |

## 常见问题

- **`import torch_npu` 报错**:torch_npu 与 torch 版本不匹配,或 CANN 未 source。
- **裸环境缺运行时依赖**(2026-09 在新容器实测,torch_npu import 链需要):
  `uv pip install pyyaml decorator attrs psutil scipy pybind11`。
- **`invalid device ordinal`**:`torch.npu.is_available()` 返回 False,检查 `npu-smi info` 是否可见设备。
- **`no member named 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE'`**(triton-ascend 3.2.0 + CANN 9.0.0):
  CANN 9.0.0 把该 enum 重命名为 `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`,而 triton-ascend 3.2.0 的
  `backends/ascend/npu_utils.cpp` 仍用旧名。修复:把 venv 里
  `triton/backends/ascend/npu_utils.cpp` 第 321 行的 `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 全部替换为
  `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`,再清空 `~/.triton` 缓存重跑。
- **RoPE kernel 编译报 `InterleaveStatusWithMaskOptimization` 断言**(triton-ascend 3.2.0):
  stride-2 交错访存 (`2 * idx` load/store) 触发后端 InterleaveOptimization pass 崩溃。
  解法:`rope_triton.py` 的 kernel 内改用**半维拆分布局**(前半/后半两次连续 load),
  wrapper 用 torch view 做 interleaved ↔ half-split 转换 —— 见 `src/rope_triton.py` 头注释。
- **RoPE 明显偏慢**:确认用 `tables=` 传预构建的 (cos, sin) NPU 张量;默认路径每次调用
  都在 host 现算三角表 + H2D,实测占比 >95%(见上文实测对比)。
- **编译很慢**:首次 `@triton.jit` 调用会触发完整编译,后续走缓存(`~/.triton/cache`)。
- **精度误差大**:确认累加器是 `tl.float32`,BLOCK 是 16 的倍数。
