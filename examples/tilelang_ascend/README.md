# tilelang_ascend/ — TileLang on Ascend NPU

本目录用 **TileLang**(北大开源的分块 kernel DSL)+ **tilelang-ascend**(昇腾后端)实现 GEMM。
TileLang 基于 TVM,用 Pythonic 语法显式描述 tiling / 数据搬运 / Ascend 内存层次调度,再编译到 Ascend。

## TileLang-Ascend 是什么

[TileLang](https://github.com/tile-ai/tilelang) 是北京大学杨智团队开发的 AI 算子编程语言,
设计目标是**让高性能 kernel 更容易写**。它采用"分块 (tiled) 编程模型":
开发者显式描述如何切分矩阵、如何搬运数据到片上缓存、如何调度流水线,编译器负责生成底层硬件代码。

**tilelang-ascend** 是 TileLang 对接华为 Ascend NPU 的后端,把 TileLang IR 编译成
AscendNPU IR / Ascend C,再经 CANN 工具链生成可执行码。

### 与其他 DSL 的区别

| 维度 | ascend_c | triton-ascend | TileLang (本目录) |
|---|---|---|---|
| 语言 | C++ | Python (`@triton.jit`) | Python (`@tilelang.jit`) |
| 底层 | bisheng 直编 | MLIR/LLVM | TVM |
| tiling 控制 | 完全手动 | 半自动 (块指针) | **显式描述** (T.copy/serial) |
| 搬运调度 | 手动 | 编译器隐式 | **显式** (alloc_L1/L0C) |
| Cube 调用 | 手动接口 | tl.dot 自动 | T.gemm_v0(init=...) |
| 抽象层级 | 最低 | 中 | 中 (偏调度) |

> TileLang 的特色是把 **Ascend 的 L1/L0C 内存层次与 Cube 调度** 显式写在代码里,
> 对硬件控制力强,适合精细调优。

## 工具链安装

### 前置:CANN 环境 (本机 CANN 9.0.0, 满足 ≥8.3.RC1 要求)
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 安装 tilelang-ascend (预编译 wheel, 推荐)

tilelang-ascend 官方提供按 CANN 版本预编译的 wheel (含 ascend 后端, 以 `tilelang` 包名发布)。
下载地址见 [releases](https://github.com/tile-ai/tilelang-ascend/releases), 选匹配
`cann版本 + 架构(aarch64/x86_64) + python版本(cp311)` 的 wheel。

```bash
cd tilelang_ascend
uv venv --python 3.11
uv sync                                          # numpy + torch (PyPI)

# torch_npu: 必须 pin 2.8.0rc1 与 torch 2.8.0 匹配!
# (2026-09 实测: 不 pin 会被连带解析成 torch-npu 2.12.0, 与 torch 2.8 符号不匹配
#  → import 报 "undefined symbol: c10::TensorImpl::decref_pyobject")
uv pip install "torch-npu==2.8.0rc1" --prerelease=allow

# tilelang-ascend 预编译 wheel (cann900 + aarch64 + cp311)
curl -L -o /tmp/tilelang.whl \
  "https://github.com/tile-ai/tilelang-ascend/releases/download/v0.1.1.010-release/tilelang-0.1.1.10%2Bubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl"
# 注意: wheel 文件名不能改 (uv/pip 校验命名规范)
uv pip install "/tmp/tilelang-0.1.1.10+ubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl"

# torch_npu 运行时依赖 (裸环境必装, 2026-09 新容器实测)
uv pip install pyyaml decorator attrs psutil scipy
```

### 验证安装
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
ACL_OP_INIT_MODE=1 uv run python -c "import torch, torch_npu, tilelang; \
    print(tilelang.__version__, torch.npu.is_available())"
```

## 编译通路

```
gemm_tilelang.py (@tilelang.jit)
   │  [TileLang 前端] Python → TileLang IR (TVM IR)
   ▼
TileLang IR (含 tiling / 搬运 / Cube 调度)
   │  [tilelang-ascend 后端] lowering → AscendNPU IR / Ascend C
   ▼
AscendNPU IR
   │  [CANN 工具链] 链接 → 可执行 kernel
   ▼
NPU Cube 单元执行 (T.gemm_v0 → Cube 16×16 矩阵乘)
```

## kernel 设计要点(见 `src/gemm_tilelang.py` 注释)

- **`is_npu=True`**:`T.Kernel(..., is_npu=True)` 声明这是 NPU kernel(走 ascend 后端,而非 GPU)。
- **Ascend 内存层次**:
  - `T.alloc_L1`:L1 buffer(Cube 核片上缓存,类比 GPU shared memory),存 A/B 子块。
  - `T.alloc_L0C`:L0C buffer(Cube 核累加器寄存器,类比 GPU fragment),存 fp32 中间结果。
- **`T.Scope("C")`**:Cube 核执行域。NPU 的 AI Core 分 Cube 核(矩阵乘)与 Vector 核(向量),
  本 GEMM 只用 Cube,故整体包在 `Scope("C")` 里。
- **`T.gemm_v0(A_L1, B_L1, C_L0, init=...)`**:Cube 单元矩阵乘。`init=True` 清零累加器,
  `False` 累加(Ascend Cube 标准累加语义,避免显式 clear)。
- **`T.copy(GM, L1)`**:GM → L1 块搬运(高效 DMA)。
- **`T.barrier_all()`**:片内同步(MTE2→MTE1 队列间)。
- **混合精度**:输入 fp16(Cube 原生),累加器 fp32。

### 优化方向(注释中讲解)

- 手动调 `block_M/N/K_L1` 匹配 L1/UB 容量。
- 多核并行:`core_num` + `T.use_swizzle` 把输出块切给多个 Cube 核。
- 软件流水:`T.set_flag`/`T.wait_flag` 手动多级流水(见官方 `example_gemm_intrinsic.py`)。
- Cube/Vector 协同:`T.set_cross_flag`/`T.wait_cross_flag`。

## 如何运行

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd tilelang_ascend
export ACL_OP_INIT_MODE=1            # 必须, 见下节 TVM FFI 冲突
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_gemm.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_rmsnorm.py
ASCEND_RT_VISIBLE_DEVICES=2 uv run python src/test_rope.py
```

### RMSNorm / RoPE 性能基准

```bash
ACL_OP_INIT_MODE=1 ASCEND_RT_VISIBLE_DEVICES=2 \
  uv run python src/bench_tilelang_ops.py --rows 16,64,256 --dims 512,4096
```

2026-09-05 实测(910B2):RMSNorm 256×4096 = 1.95 ms、RoPE 256×4096 = 10.7 ms。
注意教学版是 "一个 AI Core 处理一行 + Python 循环逐行调用" —— **每次调用的
Python/launch 开销主导耗时**, 数据反映的是本教学实现的真实成本, 与 triton 版
(一次 launch 覆盖全部行) 不直接可比。

## 文件说明

| 文件 | 作用 |
|---|---|
| `src/gemm_tilelang.py` | `@tilelang.jit` GEMM kernel (Ascend API: alloc_L1/L0C + T.gemm_v0) + `gemm()` 封装 |
| `src/softmax_tilelang.py` | 1D 行 softmax 教学 kernel (T.serial 四阶段; 0.1.13 字符串注解风格, 待迁移) |
| `src/rmsnorm_tilelang.py` | RMSNorm 2D kernel: T.tile.cast/mul + T.reduce_sum + T.tile.rsqrt/broadcast (Vector 内建指令) |
| `src/rope_tilelang.py` | RoPE 2D kernel: (cid,vid) 双 Vector 核每核一行, T.serial 逐对 fp32 旋转 |
| `src/test_gemm.py` | numpy 参考 + torch_npu 驱动 + allclose 校验 (含 ACL_OP_INIT_MODE 处理) |
| `src/test_rmsnorm.py` / `src/test_rope.py` | 多 D 档位正确性测试 (atol=1e-2, rtol=1e-2) |
| `src/bench_tilelang_ops.py` | RMSNorm/RoPE 微基准 (per-row 循环口径) |
| `pyproject.toml` | uv 配置 (numpy/torch==2.8.0); torch_npu/tilelang-ascend 手动装 |

## ⚠️ TVM FFI 冲突与 ACL_OP_INIT_MODE

tilelang-ascend 自带的 TVM 与 CANN 的 `te` 模块**共享 TVM FFI 全局注册表**, 会互相覆盖:

- 先 import tilelang → tilelang 的 TVM 注册 → torch_npu 初始化时找不到 CANN 的 `cce.product_init`
- 先 import CANN te → CANN 的 TVM 注册 → tilelang 加载 `libtilelang_module.so` 时符号缺失

**解决**:`test_gemm.py` 顶部设置 `os.environ["ACL_OP_INIT_MODE"] = "1"`(在 import torch_npu 之前)。
该变量跳过 torch_npu 的 TBE/GE 算子编译器初始化 —— 本测试只做张量分配 + tilelang 自管 kernel launch,
不走 torch_npu 图编译, 故可安全跳过, 避免冲突。

## 常见问题

- **`Cannot find global function cce.product_init`**:tilelang 的 TVM 与 CANN 的 te 冲突。设 `ACL_OP_INIT_MODE=1`(见上节)。
- **`undefined symbol ... IRBuilderFrameNode`**:CANN 的 TVM 先加载, 与 tilelang 的 TVM 不兼容。同样用 `ACL_OP_INIT_MODE=1` 跳过 torch_npu TBE 初始化。
- **`target ascend_npu not found`**:装了 PyPI 的 CUDA 版 `tilelang` 而非 ascend wheel。卸载后装预编译 cann900 wheel。
- **`CANN 版本不匹配`**:tilelang-ascend 需 CANN ≥ 8.3.RC1, 本机 9.0.0 满足; wheel 要选匹配的 cannXXX 后缀。
- **`expected Object but got str` / 注解报错**(0.1.1.10 wheel):`@T.prim_func` 参数注解必须写**直接表达式**
  `X: T.Tensor((D,), dtype)`(闭包变量在 def 时求值), 不能写 `from __future__ import annotations` 下的
  字符串前向引用 `"T.Tensor((D,), dtype)"` —— packed_func 只收 Object 不收 str。
- **`half precision operation is not allowed in aicore function`**:kernel 内 fp16 **标量**算术被禁止
  (RoPE 踩坑)。 loads 后显式 `.astype("float32")` 做乘加, 存回前再 `.astype("float16")`。
- **`Unsupported tensor type: numpy.ndarray`**:tilelang-ascend 的 cython adapter 只认 **torch NPU 张量**;
  封装器负责 numpy ↔ torch 转换 (见 `rmsnorm_tilelang.py` / `rope_tilelang.py`)。
- **编译慢**:首次 `compile()` 触发完整编译, 后续走缓存 (`~/.tilelang/cache`)。
- **精度误差大**:确认 `accum_dtype="float"`, block 是 16 的倍数。
