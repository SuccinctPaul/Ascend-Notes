# ascend-handbook

**Ascend NPU 算子开发手册** —— 从硬件到前沿，从零到一。

在 Ascend NPU 上,以 **GEMM (C = A×B)** 为案例,用 4 种不同 DSL 实现 kernel 并对比学习。
每种 DSL 单独成目录,带详细中文注释与 README,讲述该 DSL 的工具链与运行方式。

> 免责声明：本项目为个人学习笔记，与华为无隶属关系；Ascend 及相关名称为华为商标。

## 四种 DSL 目录

| 目录 | DSL | 语言 | 抽象层级 | 工具链 | 说明 |
|---|---|---|---|---|---|
| [`examples/python/`](examples/python/) | NumPy | Python | 最高 (无 NPU) | numpy + uv | **正确性基准 (ground truth)**, CPU 参考实现 |
| [`examples/ascend_c/`](examples/ascend_c/) | Ascend C | C++ | 最低 | CANN `ascendc.cmake` + `bisheng` + ACL | CANN 原生 kernel, 直接操作硬件资源 |
| [`examples/triton_ascend/`](examples/triton_ascend/) | Triton | Python (`@triton.jit`) | 中 (块级) | triton-ascend 后端 + torch_npu | OpenAI Triton 的昇腾后端, `tl.dot`→Cube |
| [`examples/tilelang_ascend/`](examples/tilelang_ascend/) | TileLang | Python (`@tilelang.jit`) | 中 (偏调度) | tilelang + tilelang-ascend 后端 | 北大开源, 显式 L1/L0C tiling + T.gemm_v0→Cube |

## 算子覆盖 (2026-09-05 实测)

当前 5 个算子 × 4 种 DSL 全部跑通并通过正确性校验(服务器:`vllm-hust-cyj-21rc-cloud-container-86`,Ascend 910B2 + CANN 9.0.0):

| 算子 | python 基准 | triton_ascend | tilelang_ascend | ascend_c | docs |
|---|---|---|---|---|---|
| GEMM (`C=A@B`) | ✅ | ✅ 0.79 ms (128³) | ✅ 0.38 ms (128³) | ✅ | [ops/09](docs/pages/ops/09-gemm.md) |
| Softmax (行归一化) | ✅ | ✅ 9/9 用例 | ✅ (见 docs §8 备注) | ✅ 4/4 用例 | [ops/03](docs/pages/ops/03-softmax.md) |
| GELU (逐元素激活) | ✅ | ✅ | ✅ | ✅ | [ops/05](docs/pages/ops/05-gelu.md) |
| RMSNorm (归一化) | ✅ | ✅ 8/8 用例 | ✅ 5/5 用例 | ✅ err=0 (16×512) | [ops/02](docs/pages/ops/02-rmsnorm.md) |
| RoPE (旋转位置编码) | ✅ | ✅ 7/7 用例 | ✅ 6/6 用例 | ✅ err=0 (16×128) | [ops/04](docs/pages/ops/04-rope.md) |

RMSNorm / RoPE 的实现说明、正确性与性能实测数据见 docs 对应页面的"本仓库实现与实测"章节。

## 实测结果 (Ascend 910B2 + CANN 9.0.0, GEMM 128³)

| DSL | NPU run | max_abs_error | 耗时 (128³ fp16) | 状态 |
|---|---|---|---|---|
| python (CPU 基准) | — | 0.0 (vs np.matmul) | 4.27 s (朴素三重循环) | ✅ PASS |
| triton_ascend | ✅ 跑通 | 0.0 | 0.79 ms | ✅ PASS |
| tilelang_ascend | ✅ 跑通 | 9.77e-04 | 0.38 ms | ✅ PASS |
| ascend_c | ✅ 跑通 | 0.0 | — | ✅ PASS |

> ascend_c 用官方 `ascendc.cmake` 框架 (`ascendc_library STATIC`) 自动完成 bisheng 编译 →
> host stub 生成 → `ascendc_pack_kernel` 打包,host 调用 `aclrtlaunch_gemm_kernel()` 启动 kernel。

## 统一约定

- **GEMM**:`C = A @ B`,`A∈R^{M×K}`,`B∈R^{K×N}`,`C∈R^{M×N}`(测试规模 M=N=K=128)。
- **RMSNorm**:`y = x / rms(x) · gamma`,`rms = sqrt(mean(x²)+eps)`,eps=1e-6,对最后一维归一化。
- **RoPE**:交错配对 (interleaved, RoFormer 原版) `pair_a=(x[2a], x[2a+1])`,θ_a=base^(-2a/d),base=10000;cos/sin 表 host 预计算,kernel 查表。
- **数据精度**:输入/输出 **float16**(Cube/Vector 原生精度);归约/中间量 **float32**(混合精度,"存窄算宽")。
- **实现层级**:朴素版为主 + 注释/README 讲解优化方向(tiling/Cube/UB/流水线)。
- **正确性校验**:每个 DSL 的 kernel 输出都与参考基准对齐,`allclose(atol=1e-2, rtol=1e-2)`,打印 PASS/FAIL。
- **包管理**:每个 Python DSL 目录用 [uv](https://github.com/astral-sh/uv) 独立 venv(`pyproject.toml`)。

## 快速对比:四种 DSL 写同一个 GEMM

```
examples/python/          →  三重循环 (np.matmul 基准)
examples/ascend_c/        →  GlobalTensor + 标量乘加 (逐元素读 GM, 最朴素)
examples/triton_ascend/   →  make_block_ptr + tl.dot (分块 + Cube 自动调用)
examples/tilelang_ascend/ →  alloc_L1/L0C + T.copy + T.gemm_v0 + T.Scope("C") (显式 Ascend 内存层次/Cube 调度)
```

## 运行环境

远程服务器:`ssh vllm-hust-cyj-21rc-cloud-container-86`,开发路径 `/root/Ascend-Notes`。
- 架构:aarch64 (Ubuntu)
- CANN:9.0.0
- NPU:Ascend910B 系列
- 所有 NPU kernel 在此服务器上构建与测试;`examples/python/` 基准可在任意机器跑。

### 运行前置(每次 shell 都要先 source)
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

## 如何逐个运行

```bash
# 0. python 基准 (CPU, 任意机器; gemm/softmax/gelu/rmsnorm/rope)
cd examples/python && uv sync && uv run python src/gemm.py src/softmax.py src/gelu.py src/rmsnorm.py src/rope.py

# 1. ascend_c (需 CANN + NPU)
cd examples/ascend_c && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/ascend_gemm            # GEMM
./build/ascend_softmax 16 512  # Softmax
./build/ascend_rmsnorm 16 512  # RMSNorm
./build/ascend_rope 16 128     # RoPE

# 2. triton_ascend (需 torch_npu + triton-ascend, 详见其 README)
cd examples/triton_ascend && ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_gemm.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_softmax.py src/test_rmsnorm.py src/test_rope.py

# 3. tilelang_ascend (需 tilelang-ascend wheel, 详见其 README)
cd examples/tilelang_ascend && ACL_OP_INIT_MODE=1 uv run python src/test_gemm.py
ACL_OP_INIT_MODE=1 uv run python src/test_rmsnorm.py src/test_rope.py
```

NPU 用例建议带 `ASCEND_RT_VISIBLE_DEVICES=<空闲卡号>` 指定设备。每个 DSL 的预期结果均包含 `PASS`。详见各目录 README。

## Docs

文档站基于 [Vocs](https://vocs.dev) 构建，源文件在 `docs/pages/`：

```bash
# 1. 安装依赖
npm install

# 2. 启动本地开发服务器（热刷新）
npm run dev

# 3. 构建静态站点（输出 dist/public/）
npm run build

# 4. 文档 MDX 语法检查（构建前防线）
node scripts/check-mdx.mjs
```

推送 `main` 分支后 GitHub Actions 自动构建并发布到 GitHub Pages（见 `.github/workflows/deploy-docs.yml`）。