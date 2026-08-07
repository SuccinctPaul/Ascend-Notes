# tilelang_ascend/ — TileLang on Ascend NPU

本目录用 **TileLang**(北大开源的分块 kernel DSL)+ **tilelang-ascend**(昇腾后端)实现 GEMM。
TileLang 基于 TVM,用 Pythonic 语法显式描述 tiling / 数据搬运 / 流水线,再编译到 Ascend。

## TileLang 是什么

[TileLang](https://github.com/tile-ai/tilelang) 是北京大学杨智团队主导开发的 AI 算子编程语言,
设计目标是**让高性能 kernel 更容易写**。它采用"分块 (tiled) 编程模型":
开发者显式描述如何切分矩阵、如何搬运数据到片上缓存、如何调度流水线,
编译器负责生成底层硬件代码。

**tilelang-ascend** 是 TileLang 对接华为 Ascend NPU 的后端,把 TileLang IR 编译成
AscendNPU IR,再经 CANN 工具链生成可执行码。

### 与其他 DSL 的区别

| 维度 | ascend_c | triton-ascend | TileLang (本目录) |
|---|---|---|---|
| 语言 | C++ | Python (`@triton.jit`) | Python (`@tilelang.jit`) |
| 底层 | TVM | MLIR/LLVM | TVM |
| tiling 控制 | 完全手动 | 半自动 (块指针) | **显式描述** (T.copy/Pipelined) |
| 搬运调度 | 手动 | 编译器隐式 | **显式** (alloc_shared/Pipelined) |
| 抽象层级 | 最低 | 中 | 中 (偏调度) |

> TileLang 的特色是把 **数据搬运与流水线** 显式写在代码里,对 Ascend 的 UB/L1 内存层次
> 控制力强,适合精细调优。

## 工具链安装

### 前置:CANN 环境 (本机 CANN 9.0.0, 满足 ≥8.2.RC1 要求)
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 安装 tilelang + tilelang-ascend 后端
```bash
cd tilelang_ascend
uv venv --python 3.11
uv sync                         # 安装 numpy, torch

# 1) tilelang 主包
uv pip install tilelang

# 2) tilelang-ascend 后端 (含 Ascend NPU IR 适配)
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git /tmp/tilelang-ascend
cd /tmp/tilelang-ascend
bash install_ascend.sh
source set_env.sh               # 注册 ascend 后端到 tilelang
```

### 验证安装
```bash
uv run python -c "import tilelang; print(tilelang.__version__)"
```

## 编译通路

```
gemm_tilelang.py (@tilelang.jit)
   │  [TileLang 前端] Python → TileLang IR (TVM IR)
   ▼
TileLang IR (含 tiling / 搬运 / 流水线调度)
   │  [tilelang-ascend 后端] lowering → AscendNPU IR
   ▼
AscendNPU IR
   │  [CANN 工具链] 链接 → 可执行 kernel
   ▼
NPU Cube/Vector 单元执行
```

## kernel 设计要点(见 `src/gemm_tilelang.py` 注释)

- **显式分块**:`T.Kernel(ceildiv(N,block_N), ceildiv(M,block_M))` 声明并行维度。
- **片上缓冲**:`T.alloc_shared`(对应 Ascend L1/UB)、`T.alloc_fragment`(寄存器累加器)。
- **显式搬运**:`T.copy(GM, shared)` —— 块搬运,对应高效 DMA。
- **`T.gemm`**:块矩阵乘原语,ascend 后端映射到 Cube 单元。
- **`T.Pipelined(..., num_stages=3)`**:3 级流水线,搬运与计算重叠。
- **混合精度**:输入 fp16,累加器 fp32。

### 等价的底层展开(注释中讲解)

`T.gemm` + `T.copy` 在 ascend 后端会展开为类似:
```
for ko in range(0, K, block_K):
    alloc_ub(A_tile); copy_gm_to_ub(A_tile, A, mo, ko)   # GM -> UB (DMA)
    alloc_ub(B_tile); copy_gm_to_ub(B_tile, B, ko, no)
    vector_mad(C_tile, A_tile, B_tile)                    # Cube 乘加
copy_ub_to_gm(C, C_tile, mo, no)                          # UB -> GM 写回
```

### 优化方向

- 手动调 `block_M/N/K` 匹配 UB 容量。
- 跨核同步:`T.set_cross_flag` / `T.wait_cross_flag`(多核 tiling)。
- 调 `num_stages` 平衡流水线深度与寄存器压力。

## 如何运行

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# 若装了 tilelang-ascend, 还需 source 它的 set_env.sh
cd tilelang_ascend
uv run python src/test_gemm.py
```

预期输出:
```
[INFO] === TileLang-Ascend GEMM 测试 (dtype=float16) ===
[INFO] 预热编译 (首次调用触发 tilelang-ascend 编译)...
[INFO] TileLang kernel 耗时: X.XXXX ms
[INFO] 校验结果: PASS (max_abs_error=..., atol=1e-2, rtol=1e-2)
[INFO] TileLang-Ascend GEMM 测试完成, 全部 PASS
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `src/gemm_tilelang.py` | `@tilelang.jit` GEMM kernel (显式 tiling + 流水线) + `gemm()` 封装 |
| `src/test_gemm.py` | numpy 参考 + torch_npu 驱动 + allclose 校验 |
| `pyproject.toml` | uv 配置 (numpy/torch); tilelang/tilelang-ascend 手动装 |

## 常见问题

- **`target ascend_npu not found`**:tilelang-ascend 后端未注册,确认 `install_ascend.sh` 已跑 + `source set_env.sh`。
- **`CANN 版本不匹配`**:tilelang-ascend 需 CANN ≥ 8.2.RC1,本机 9.0.0 满足。
- **编译慢**:首次 `compile()` 触发完整编译,后续走缓存。
- **精度误差大**:确认 `accum_dtype=T.float32`,block 是 16 的倍数。
