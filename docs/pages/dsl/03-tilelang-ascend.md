# 03 · TileLang on Ascend 核心手册

> 面向 0 到 1 新手的「昇腾 NPU 四种 DSL」第三篇。今天我们回答一个问题：
> **如果 Triton 的"块级 + 编译器自动搬运"还不够、想自己控制每一级片上缓冲和 Cube 调用，该用哪条路？** 答案就是 **TileLang + tilelang-ascend**。

***

## TL;DR

**TileLang** 是北京大学杨智团队开源的 **分块(tiled)kernel DSL**,基于 TVM,用 Pythonic 语法
**显式**描述 tiling、数据搬运、Ascend 片上内存层次(L1 / L0C / UB)和 Cube/Vector 调度。
**tilelang-ascend** 是它对接华为 Ascend NPU 的后端,把 TileLang IR 编译成
**AscendNPU IR / Ascend C**,再经 CANN 工具链生成可执行 kernel。

在四种 DSL 的抽象梯子里,TileLang 位于 **Triton 之下、Ascend C 之上**:比 Triton 多了
"显式指定数据搬进 L1 还是 L0C、显式调用 `T.gemm_v0` 喂 Cube"的控制力;比 Ascend C 少一层
"裸 C++ + bisheng 直编"的复杂度。本仓库实测 128³ fp16 GEMM:**0.38 ms**(Ascend 910B2 +
CANN 9.0.0),是该规模四种 DSL 里最快的写法。

| 维度          | TileLang (tilelang-ascend)                                                             |
| ----------- | -------------------------------------------------------------------------------------- |
| **语言**      | Python (`@tilelang.jit` + `T.prim_func`)                                               |
| **底层 IR**   | TVM / TensorIR(TLang 自有前端,复用 TVM stack)                                                |
| **后端**      | tilelang-ascend → AscendNPU IR / Ascend C → CANN                                       |
| **抽象层级**    | 中等偏上(调度级,显式 L1/L0C)                                                                    |
| **关键原语**    | `T.alloc_L1` / `T.alloc_L0C` / `T.alloc_ub` / `T.copy` / `T.gemm_v0` / `T.barrier_all` |
| **Cube 调用** | `T.gemm_v0(A_L1, B_L1, C_L0, init=...)` 显式                                             |
| **搬运控制**    | 显式:`T.copy(GM↔L1)` / `T.copy(L0C↔GM)` / `T.copy(GM↔UB)`                                |
| **精度策略**    | fp16 输入/输出 + fp32 累加器(混合精度,Cube 原生)                                                    |
| **本仓库实测**   | 128³ GEMM 0.38 ms,9.77e-04 误差,PASS                                                     |
| **典型场景**    | 需要精确控制 L1/L0C 分配与流水线调度的高性能 kernel                                                      |

> **人话**:TileLang 让你用 Python 写 kernel,但"搬进哪级缓存、何时同步、Cube 何时算"
> 全是你说了算——不是编译器替你决定。控制力比 Triton 强,写起来比 Ascend C 容易。

***

## Background

### TileLang 的起源:北大杨智团队 + TVM

[TileLang](https://github.com/tile-ai/tilelang) 由北京大学杨智团队开发并开源,设计目标是
**让高性能 kernel 更容易写**。它采用"分块(tiled)编程模型":开发者显式描述如何切分矩阵、
如何搬运数据到片上缓存、如何调度流水线,编译器负责生成底层硬件代码。

TileLang **基于 TVM**:前端把自己的 Python DSL(`@tilelang.jit` + `T.prim_func`)解析成
TVM IR(TensorIR),复用 TVM 的 schedule / lower / codegen 通路。TileLang IR 本身带有
tiling、搬运、Cube 调度等高层语义,这些在普通 TVM TensorIR 里是没有的——这正是它"夹层"
地位的来源。

### tilelang-ascend 后端:TileLang IR → AscendNPU IR / Ascend C → CANN 可执行码

**tilelang-ascend** 是 TileLang 对接华为 Ascend NPU 的后端,编译通路:

```
TileLang IR (含 tiling / 搬运 / Cube 调度)
   │  [tilelang-ascend 后端] lowering
   ▼
AscendNPU IR / Ascend C
   │  [CANN 工具链] 链接
   ▼
NPU 可执行 kernel (跑在 AI Core 的 Cube/Vector 单元)
```

后端把 TileLang 的 `T.alloc_L1` / `T.alloc_L0C` / `T.copy` / `T.gemm_v0` 等高层原语,逐个
lower 成 AscendNPU IR 上的 buffer 分配 + DMA + Cube 矩阵乘调用,最终交给 CANN 工具链产出
可加载的 `.o` / kernel module。

### 与 TVM / TensorIR 的关系

- TileLang **复用** TVM 的 IR 基础设施(TensorIR / Pass / Build),但前端语法是自己的。

- 开发者**不直接写** TVM的 `te.compute` / `tir.schedule`,而是写 `T.copy` / `T.gemm_v0`
  这种 TileLang 自有原语,再由 tilelang-ascend 后端翻译成 AscendNPU IR。

- 换句话说,TileLang = "在 TVM 之上加了一层**带硬件层级语义的 tiled DSL**"。

### 与 Triton 的定位差异:偏调度 vs 偏块级

| 维度          | Triton (triton-ascend) | TileLang (tilelang-ascend)        |
| ----------- | ---------------------- | --------------------------------- |
| **核心抽象**    | 块(block),编译器自动决定搬运     | tiling + 搬运 + Cube 调度,全部显式        |
| **缓冲分配**    | 隐式(编译器决定)              | `T.alloc_L1` / `T.alloc_L0C` 显式   |
| **Cube 调用** | `tl.dot` → 自动映射        | `T.gemm_v0(...)` 显式调用             |
| **搬运语义**    | `tl.load` / `tl.store` | `T.copy(GM↔L1)` + `T.barrier_all` |
| **累加器控制**   | 编译器管                   | `init=(k == 0)` 清零 / `False` 累加   |
| **抽象层级**    | 中(块级)                  | 中低(调度级)                           |
| **底层 IR**   | MLIR/LLVM              | TVM                               |

> **人话**:Triton 是"说清楚块大小,编译器帮你搬";TileLang 是"我来指定搬进 L1 还是 L0C,
> Cube 何时算、何时同步"。两者都跑在 AI Core 上,差的是"谁决定搬运"。

***

## Why

### 为什么用 TileLang

NPU 上一个高性能 kernel 的性能,几乎全部来自**数据怎么搬、算力怎么喂**——而不是数学公式。
Ascend 的 L1 / L0C / UB 是**软件显式管理**的(没有硬件 cache 自动补货),你不搬数据就不来。

Triton 把这层"搬运调度"交给编译器自动决策,在多数场景下足够好,但:

- 当你**明确知道**这份数据该进 L1 还是 UB;

- 当你想**手动做双缓冲 / 软件流水**让 MTE2/MTE1 队列充分并行;

- 当你想**精确控制**累加器是 fp32 还是 fp16、第一个 K 块清零后续累加……

Triton 没法给你这些旋钮,而 TileLang 把它们**显式写在代码里**。tilelang-ascend 后端会忠实地
按你写的搬运 + 同步 + Cube 调度生成 IR,而不是"自己想一套"。这就是它**控制力最强**的来源。

### 什么场景适合

- **需要精确控制 L1 / L0C 分配**的 GEMM / Conv / attention 类 kernel。

- **需要显式软件流水线**(`T.set_flag` / `T.wait_flag` 多级流水)的场景。

- **需要 Cube + Vector 协同**(`T.set_cross_flag` / `T.wait_cross_flag`)的混合 kernel。

- **学习 / 教学 Ascend 调度细节**——TileLang 的 Pythonic 语法 + 显式原语,是最适合
  在 Python 里"看见" L1/L0C/UB 流向的 DSL。

不适合:

- **只是想跑通**一个 op、不在意极致性能:Triton 一行 `tl.dot` 就够,别上 TileLang。

- **需要每字节级控制**(寄存器分配、instrinsic 指令):用 Ascend C。

### 抽象梯子:比 Triton 高一层控制力,比 Ascend C 低一层抽象

```mermaid
flowchart LR
    subgraph LADDER["抽象梯子(从高到低)"]
        direction LR
        PY["Python/NumPy\nCPU 基准"]
        TR["Triton\n块级 + 编译器搬运"]
        TL["TileLang\n显式 L1/L0C + Cube"]
        AC["Ascend C\nC++ 全手动"]
    end
    PY -->|"抽象降低"| TR
    TR -->|"+L1/L0C 显式"| TL
    TL -->|"−Python +裸 C++"| AC
```

> **人话**:Python 是"只管对不对",Triton 是"说清块大小、编译器搬",TileLang 是"我指定
> 搬进哪级缓存、Cube 怎么算",Ascend C 是"每一搬每算都我亲手写"。

***

## 正文

### 1. 工具链全景:从 `@tilelang.jit` 到 NPU 可执行 kernel

```mermaid
flowchart TB
    SRC["gemm_tilelang.py\n@tilelang.jit(out_idx=[-1])\n@T.prim_func"] --> FE["TileLang 前端\nPython → TileLang IR\n(TVM IR + tiled 语义)"]
    FE --> IR["TileLang IR\n含 tiling / 搬运 / Cube 调度\nT.alloc_L1 / T.gemm_v0 / T.barrier_all"]
    IR --> BE["tilelang-ascend 后端\nLowerTileOp + CodeGenTileLangAscend"]
    BE --> ASC_IR["AscendNPU IR / Ascend C\nbuffer 分配 + DMA + Cube MatMul"]
    ASC_IR --> CANN["CANN 工具链\n链接 → 可执行 kernel (.o / module)"]
    CANN --> NPU["NPU AI Core 执行\nCube 16×16 MAC 阵列\nMTE2/MTE1/MTE3 队列"]
```

> **人话**:`@tilelang.jit` 装饰的 Python 函数,先被 TileLang 前端翻译成带 tiling 语义的
> TVM IR,再由 tilelang-ascend 后端 lower 成 AscendNPU IR,最后 CANN 链接成可执行码,
> 跑在 AI Core 的 Cube/Vector 上。整条链路一次完成,首次调用触发、后续走缓存。

### 2. 安装与环境

#### 前置:CANN 环境(≥ 8.3.RC1,本机 9.0.0)

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

#### 安装 tilelang-ascend(预编译 wheel,推荐)

tilelang-ascend 官方提供按 CANN 版本预编译的 wheel(以 `tilelang` 包名发布,内含 ascend 后端)。版本号在源码注释里写作 `v0.1.1.010`、在 wheel 文件名里写作 `0.1.1.10+ubuntu.20.4.cann900`,是**同一发行版的两种写法**,本文统一称 `v0.1.1.010`。
下载地址见 [releases](https://github.com/tile-ai/tilelang-ascend/releases),选匹配
`cann版本 + 架构(aarch64/x86_64) + python 版本(cp311)` 的 wheel。

```bash
cd tilelang_ascend
uv venv --python 3.11
uv sync                                          # numpy + torch (PyPI)

# torch_npu (匹配 torch 2.8.0,从昇腾官方获取 cp311 aarch64 wheel)
uv pip install torch_npu-2.8.0rc1-cp311-cp311-manylinux_2_28_aarch64.whl

# tilelang-ascend 预编译 wheel (本机用 cann900 + aarch64 + cp311)
uv pip install tilelang-0.1.1.10+ubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl

# pyyaml (torch_npu 运行时依赖,不在 pyproject 里)
uv pip install pyyaml
```

> **重要**:PyPI 上的 `tilelang` 主包(如 0.1.13)是 **CUDA 版,不含 ascend 后端**。
> 必须装 tilelang-ascend 的预编译 wheel(或源码 `install_ascend.sh`),它以同名 `tilelang`
> 包覆盖安装,内含编译好的 ascend TVM 后端。

#### 验证安装

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source .venv/bin/activate
python -c "import tilelang; print(tilelang.__version__)"
```

#### pyproject.toml(本仓库的版本锁)

```toml
[project]
name = "tilelang-ascend"
version = "0.1.0"
requires-python = ">=3.11,<3.12"     # 锁 3.11:与 triton_ascend 对齐,共用 torch_npu 2.8.0rc1
dependencies = [
    "numpy>=2.0",
    "torch==2.8.0",                  # 必须与 torch_npu 2.8.0rc1 严格一致
]
# tilelang 主包 + tilelang-ascend 后端 + torch_npu 需手动安装到本 venv
# 详见 README。
```

> **TVM FFI 冲突注意**:tilelang-ascend 自带的 TVM 与 CANN 的 `te` 模块**共享 TVM FFI
> 全局注册表**,会互相覆盖。在 `import torch_npu` **之前**必须设:
>
> ```python
> import os
> os.environ.setdefault("ACL_OP_INIT_MODE", "1")  # 跳过 torch_npu 的 TBE/GE 算子编译器初始化
> ```
>
> 本测试只做张量分配 + tilelang 自管 kernel launch,不走 torch\_npu 图编译,故可安全跳过。
> 详见 `test_gemm.py` 顶部注释。

### 3. 第一个 kernel:GEMM

源码:`examples/tilelang_ascend/src/gemm_tilelang.py`。下面**逐段**讲解,对齐
Ascend 内存层级和 Cube 调度语义。

#### 3.1 装饰器:`@tilelang.jit(out_idx=[-1])`

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def gemm_matmul(M, N, K, block_M, block_N, K_L1,
                dtype="float16", accum_dtype="float"):
    ...
    return main
```

- `@tilelang.jit`:把外层 Python 函数变成一个 TileLang kernel factory。**调用时**才触发
  编译(`gemm_matmul(M, N, K, ...)` 返回一个可调用 kernel)。

- `out_idx=[-1]`:返回**最后一个张量参数**(即 `C`)作为输出。TileLang 据此自动分配输出
  张量并接住 kernel 写回。

- 外层参数 `M / N / K / block_M / block_N / K_L1 / dtype / accum_dtype` 都是**编译期常量**,
  每一组不同取值会编出不同的 kernel(走缓存复用)。

#### 3.2 `T.prim_func` 声明输入/输出张量

```python
@T.prim_func
def main(
    A: T.Tensor((M, K), dtype),    # 输入 A: (M, K) fp16
    B: T.Tensor((K, N), dtype),    # 输入 B: (K, N) fp16
    C: T.Tensor((M, N), dtype),    # 输出 C: (M, N) fp16
):
```

- `T.prim_func`:把 `main` 标记成 TileLang 的 IR 入口函数,前端会用 TVM script parser
  把它的 AST 解析成 TensorIR。

- 张量用 `T.Tensor(shape, dtype)` 注解;`shape` 和 `dtype` 用闭包外层的编译期参数填值。

- **坑**:`tilelang-ascend` v0.1.1.010 的 parser 要求注解是**实际 Buffer 对象**——
  本文件**不能**加 `from __future__ import annotations`(否则注解会变成 str,
  抛 `TVMError: expected Object but got str (type_code 11 vs 8)`,
  见下文 [§5.4 注解 workaround](#54-注解-workaroundtl-5) 和 ops/05-gelu §8.10.4 常见坑 #TL-5)。

#### 3.3 `T.Kernel(..., is_npu=True)` 声明并行维度

```python
m_num = M // block_M    # M 维 block 数
n_num = N // block_N    # N 维 block 数

with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num   # M 维 block 索引
    by = cid % n_num    # N 维 block 索引
```

- `T.Kernel(grid, is_npu=True)`:声明并行维度。

- **`is_npu=True`** **是关键**:告诉 tilelang 这是 **NPU kernel**,走 ascend 后端,而不是
  GPU 的 CUDA thread block。**不加这个参数**,会按 GPU 语义生成代码,加载时报
  `target ascend_npu not found` 或 `Cannot find global function cce.product_init`
  (见 [FAQ Q1](#q1-is_nputrue-不加会怎样))。

- 这里把 `m_num * n_num` 个输出块拍平成 1 维,由 `cid` 自行拆回 `(bx, by)`。单 block
  时 `grid=1`(单核);多 block 时多核并行。

- 解包写法跟 `is_npu` 走:**带 `is_npu=True` 时** `T.Kernel` 返回二元序列,必须 `as (cid, _)`
  (GEMM 源码如此);**不带 `is_npu` 时**单变量 `as cid` 即可(本仓库 gelu/softmax 源码如此,
  此时 ascend 后端由全局 target 自动探测,见 [ops/05 §8.10](/ops/05-gelu))。

#### 3.4 `T.alloc_L1` / `T.alloc_L0C` 显式分配片上缓冲

```python
A_L1 = T.alloc_L1((block_M, K_L1), dtype)
B_L1 = T.alloc_L1((K_L1, block_N), dtype)
C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
```

| 原语              | 对应 Ascend 缓冲                 | 类比 GPU        | 用途             |
| --------------- | ---------------------------- | ------------- | -------------- |
| `T.alloc_L1`    | **L1**(Cube/Vector 共用片上高速缓存) | shared memory | 存 A/B 子块       |
| `T.alloc_L0C`   | **L0C**(Cube 累加器寄存器,fp32)    | fragment      | 存 fp32 中间结果    |
| `T.alloc_ub`    | **UB**(Vector 工作台,逐元素算子)     | shared.dyn    | Vector 核的输入/输出 |
| `T.alloc_local` | thread-private(本地缓冲)         | local         | 标量/中间值         |

> **人话**:L1 是 Cube 核的"高速小仓",L0C 是 Cube 的"累加器寄存器"。在 TileLang 里
> 你**显式声明**这些 buffer,编译器不会替你决定——这正是它控制力强的根源。

#### 3.5 `T.Scope("C")` Cube 执行域

```python
with T.Scope("C"):
    ...
```

- NPU 的 AI Core 分 **Cube 核**(矩阵乘)和 **Vector 核**(逐元素运算)两类执行单元。

- `T.Scope("C")` 把整段代码标记为 **Cube 核执行域**,所有 `T.gemm_v0` / `T.copy` 在
  这里的语义都按 Cube 核的搬运通路走(MTE2/MTE1 队列)。

- 逐元素 kernel 不一定要显式 `T.Scope`:本仓库的 GELU/Softmax 源码就没写(后端按算子形态自动走 Vector 通路);需要手动分核调度时才用,如 GEMM 的 `T.Scope("C")`。

#### 3.6 `T.copy(GM→L1)` DMA 块搬运 + `T.barrier_all()` 同步

```python
loop_k = T.ceildiv(K, K_L1)   # K 维分块数
for k in T.serial(loop_k):
    # GM -> L1 块搬运(DMA,高效)
    T.copy(A[bx * block_M, k * K_L1], A_L1)
    T.copy(B[k * K_L1, by * block_N], B_L1)

    # 片内同步:确保搬运完成再计算(MTE2→MTE1 队列)
    T.barrier_all()
```

- `T.ceildiv(K, K_L1)`:K 维分多少块。`T.ceildiv` 是向上整除。

- `T.serial(loop_k)`:**串行**循环(不是并行)。K 维累加必须按顺序,不能跨 K 并发。

- `T.copy(GM_slice, L1_buf)`:把 GM 上的一块数据**用 DMA 搬到 L1**。这是高效的块搬运,
  对应 Ascend `DataCopy` 指令。

- `T.barrier_all()`:**片内同步**。MTE2(Move Engine 2,GM→L1 搬运队列)和
  MTE1(L1→L0A/L0B 搬运队列)是异步的,这里保证"搬完再算"。

  - 见 [FAQ Q5](#q5-tbarrier_all-的作用是什么)。

#### 3.7 `T.gemm_v0(..., init=...)` Cube 矩阵乘

```python
T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

T.barrier_all()
```

- `T.gemm_v0(A, B, C, init=...)`:**Cube 单元矩阵乘**。`A_L1 @ B_L1` 累加到 `C_L0`(L0C
  累加器,fp32)。

- **`init`** **参数语义**(详见 [FAQ Q3](#q3-tgemm_v0-的-init-参数语义是什么)):

  - `init=True` → **清零累加器再乘**(第一个 K 块用,避免残留脏数据)。

  - `init=False` → **累加**(后续 K 块用,把这次的结果加到上一次的 C\_L0 上)。

  - `init=(k == 0)` 是 Ascend Cube 的标准累加写法,**避免显式 clear**,一行代码同时
    表达"首块清零 + 后续累加"。

- 后面再 `T.barrier_all()`:保证 Cube 算完再进入下一轮 K 循环(下一轮会覆写 A\_L1 / B\_L1)。

#### 3.8 `T.copy(L0C→GM)` 写回

```python
# L0C -> GM 写回(fp32 累加结果转 fp16)
T.copy(C_L0, C[bx * block_M, by * block_N])
```

- 把 fp32 累加器 `C_L0` 一次性搬回 GM 上的输出块 `C[...]`。tilelang-ascend 后端会处理
  fp32→fp16 的精度转换。

- 至此一个输出块 `(block_M × block_N)` 完成。多核并行时,不同 `cid` 负责不同输出块。

#### 3.9 便捷封装:`gemm(a, b, block_M=128, block_N=128, K_L1=64)`

```python
def gemm(a, b, block_M=128, block_N=128, K_L1=64):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and M % block_M == 0 and N % block_N == 0
    kernel = gemm_matmul(M, N, K, block_M, block_N, K_L1)
    c = kernel(a, b)
    return c
```

- 输入 torch 张量,返回 `C = A @ B`。tilelang kernel 接受 torch tensor,ascend 后端要求
  张量在 npu 设备上。

- 首次调用会触发 tilelang-ascend 编译(生成 ascend kernel),较慢;后续相同 M/N/K + 块参数
  走 `~/.tilelang/cache` 缓存。

### 4. TileLang 核心编程模型

下面把 GEMM 里出现的原语 + 没出现但常用的原语一并钉死,配一张 ASCII 心法图。

#### 4.1 原语速查表

| 原语                                                | 作用                            | 对应 Ascend 概念   |
| ------------------------------------------------- | ----------------------------- | -------------- |
| `@tilelang.jit(out_idx=[-1])`                     | kernel factory 装饰器,声明输出张量     | —              |
| `@T.prim_func`                                    | TileLang IR 入口函数              | TVM script     |
| `T.Kernel(grid, is_npu=True)`                     | 声明并行维度 + 标记 NPU kernel        | grid / block   |
| `T.alloc_L1(shape, dtype)`                        | 分配 L1 buffer(Cube/Vector 共用)  | L1 高速缓存        |
| `T.alloc_L0C(shape, dtype)`                       | 分配 L0C 累加器(Cube 专用,fp32)      | L0C 累加器        |
| `T.alloc_ub(shape, dtype)`                        | 分配 UB(Vector 核工作台)            | Unified Buffer |
| `T.alloc_local(shape, dtype)`                     | 分配 thread-private 本地缓冲        | local          |
| `T.Scope("C")` / `T.Scope("V")` / `T.Scope("M")`  | Cube / Vector / Mixed 执行域     | Cube/Vector 核  |
| `T.copy(src, dst)`                                | DMA 块搬运(GM↔L1, L0C↔GM, GM↔UB) | DataCopy(DMA)  |
| `T.gemm_v0(A, B, C, init=...)`                    | Cube 单元矩阵乘 + 累加语义             | Cube MatMul    |
| `T.barrier_all()`                                 | 片内同步(MTE2/MTE1/MTE3 队列间)      | sync barrier   |
| `T.serial(n)` / `T.serial(start, end)`            | 串行循环(K 维累加必用)                 | for 循环         |
| `T.ceildiv(a, b)`                                 | 向上整除                          | —              |
| `T.ascend_tile.<op>(dst, src1, [src2_or_scalar])` | Vector 核 buffer 级原语           | Vector 指令      |
| `T.set_flag` / `T.wait_flag`                      | 手动多级流水线(软件流水)                 | MTE 队列控制       |
| `T.set_cross_flag` / `T.wait_cross_flag`          | Cube / Vector 协同              | 跨核同步           |
| `T.use_swizzle`                                   | 多核 swizzle(输出块切给多 Cube 核)     | 多核切分           |

#### 4.2 关键参数:`is_npu=True` 的语义

`T.Kernel(grid, is_npu=True)` 里的 `is_npu=True` 不是可选项——它告诉 tilelang:
**这是 NPU kernel,走 ascend 后端**。不加的话:

- tilelang 会按 **GPU CUDA thread block** 语义生成 IR(用 `threadIdx` / `blockIdx`)。

- 加载时找不到 `cce.product_init` 之类的 NPU 运行时符号,或者报
  `target ascend_npu not found`。

- 详见 [FAQ Q1](#q1-is_nputrue-不加会怎样)。

#### 4.3 缓冲分配:L1 / L0C / UB / local

TileLang 把 Ascend 的内存层级**显式暴露**:

- **L1**(`T.alloc_L1`):Cube/Vector 共用片上缓存。GEMM 里存 A/B 子块。

- **L0C**(`T.alloc_L0C`):Cube 专用累加器,fp32。GEMM 里存矩阵乘中间结果。

- **UB**(`T.alloc_ub`):Vector 核工作台,逐元素算子的输入/输出。GELU 里存 x / 中间结果。

- **local**(`T.alloc_local`):thread-private 本地缓冲。Softmax 里存标量 max/sum/inv。

> **坑**:tilelang-ascend v0.1.1.010 的 `AscendCopy`(DMA)只允许 **global ↔ shared**
> 或 **global ↔ shared.dyn**。把 `T.alloc_local`(scope=local)错用于 NPU Vector 核 UB,
> 会抛 `TVMError: Unsupported scope: src = global, dst = local`
> (见 ops/05-gelu §8.10.4 常见坑 #TL-2,GELU 上实测复现)。
> **正确姿势**:Vector 核 UB 用 `T.alloc_ub`(scope=shared → UB),
> Cube/L1 用 `T.alloc_L1`(scope=shared.dyn → L1)。

#### 4.4 执行域:`T.Scope("C")` / `T.Scope("V")`

NPU 的 AI Core 分 Cube 核(矩阵乘)和 Vector 核(逐元素):

- **`T.Scope("C")`**:Cube 核执行域,所有 `T.gemm_v0` 必须在 C scope 里。

- **`T.Scope("V")`**:Vector 核执行域,逐元素算子走这里(本仓库 GELU/Softmax 未显式写,后端自动处理;需要显式分核时再用)。

- Mixed scope(`T.Scope("M")`)允许 Cube + Vector 协同。

#### 4.5 搬运:`T.copy` 的三种典型路径

| 搬运路径                        | 用途                 | 典型场景     |
| --------------------------- | ------------------ | -------- |
| `T.copy(GM_slice, L1_buf)`  | GM → L1(GEMM 输入)   | 喂 Cube   |
| `T.copy(L0C_buf, GM_slice)` | L0C → GM(GEMM 输出)  | 写回累加结果   |
| `T.copy(GM_slice, UB_buf)`  | GM → UB(Vector 输入) | 喂 Vector |
| `T.copy(UB_buf, GM_slice)`  | UB → GM(Vector 输出) | 写回逐元素结果  |

每次 `T.copy` 都对应一次 DMA,搬运和计算是异步的,需要 `T.barrier_all()` 同步。

#### 4.6 Cube 调用:`T.gemm_v0` 与 `init` 语义

```python
T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
```

- `A_L1`、`B_L1` 在 L1,`C_L0` 在 L0C。Cube 单元做 `C_L0 += A_L1 @ B_L1`(或 `=` 视 init)。

- `init=True` → `C_L0 = A_L1 @ B_L1`(清零再乘)。

- `init=False` → `C_L0 += A_L1 @ B_L1`(累加)。

- `init=(k == 0)` → 首块清零、后续累加,一行表达完整 K 维累加语义。

- Cube 内部按 16×16 MAC 阵列做矩阵乘,fp16 输入、fp32 累加(混合精度,Cube 原生)。

#### 4.7 同步:`T.barrier_all` 的作用

Ascend 的 MTE2(GM→L1 搬运队列)、MTE1(L1→L0A/L0B 搬运队列)、MTE3(L0C→GM 写回
队列)和 Vector 队列**彼此异步**。`T.barrier_all()` 是一个**全队列同步 barrier**:

- 保证"GM→L1 搬完再 L1→L0A/L0B 搬"(MTE2 → MTE1)。

- 保证"L0C 算完再写回 GM"(MTE3 → 后续 MTE2)。

- 保证"下一轮 K 循环开始前,A\_L1/B\_L1 已被消费完"(避免覆写未读数据)。

详见 [FAQ Q5](#q5-tbarrier_all-的作用是什么)。

#### 4.8 循环与分块:`T.serial` + `T.ceildiv`

- `T.serial(n)`:**串行** for 循环。K 维累加必须串行(不能跨 K 块并行,否则累加结果错)。

- `T.serial(start, end)`:带起止的串行循环。

- `T.ceildiv(a, b)`:向上整除,常用来算"K 维分多少块"。

#### 4.9 ASCII 心法图:TileLang 显式调度 vs Triton 隐式调度

```
                   Triton 隐式调度                        TileLang 显式调度
                   ──────────────                         ──────────────

  你写的:          for k in range(K // BK):              for k in T.serial(loop_k):
                      A_tile = tl.load(A_p[..., k*BK])        T.copy(A[..., k*K_L1], A_L1)   ← 显式 GM→L1
                      B_tile = tl.load(B_p[k*BK, ...])        T.copy(B[k*K_L1, ...], B_L1)   ← 显式 GM→L1
                      acc    = tl.dot(A, B, acc)             T.barrier_all()                ← 显式同步
                                                            T.gemm_v0(A_L1, B_L1, C_L0,    ← 显式 Cube
                                                                      init=(k==0))
                                                            T.barrier_all()
                   tl.store(C_p[...], acc)                  T.copy(C_L0, C[..., ...])      ← 显式 L0C→GM

  缓冲位置:        编译器决定(可能进 L1, 可能进 L0C)       你指定 A_L1 @ L1, C_L0 @ L0C
  同步时机:        编译器决定                                你手写 T.barrier_all()
  累加清零:        acc 初值 0 + tl.dot(..., acc=acc)         init=(k==0) 直接交给 Cube 单元
  控制粒度:        "我把这块算出来"                          "我把这块搬进 L1, 算进 L0C, 写回 GM"
```

> **人话**:同样是"一个 GEMM block",Triton 让你只关心**块大小**,搬运和同步交给
> 编译器;TileLang 让你**亲手写**每一步搬运、同步、Cube 调用。控制力上一档,
> 心智负担也上一档。

#### 4.10 ASCII 心法图:L1 / L0C / UB 内存层级

```
   GM (HBM,容量最大,带宽有限)──────────────────────────────────────────┐
        │                                                                │
        │ DMA (T.copy GM→L1 / GM→UB)                                     │
        ▼                                                                │
   ┌─────────────────── Cube 核私有 ──────────────┐                     │
   │  L1 (高速缓存,Cube/Vector 共用)                │  ← T.alloc_L1     │
   │     ┌──────────────┐  ┌──────────────┐        │                   │
   │     │ A_L1 子块    │  │ B_L1 子块    │        │                   │
   │     └──────┬───────┘  └──────┬───────┘        │                   │
   │            │   MTE1 搬运       │  MTE1 搬运    │                   │
   │            ▼                   ▼                │                   │
   │        L0A (Cube A 输入)   L0B (Cube B 输入)    │                   │
   │            └──────────┐ ┌──────────┘           │                   │
   │                       ▼ ▼                       │                   │
   │                  ┌──────────┐                   │                   │
   │                  │  L0C     │ ← T.alloc_L0C    │                   │
   │                  │ (累加器  │   (fp32 累加)     │                   │
   │                  │  寄存器) │                   │                   │
   │                  └────┬─────┘                   │                   │
   └───────────────────────┼────────────────────────┘                   │
                           │ DMA (T.copy L0C→GM)                          │
                           ▼                                              │
   ┌─────────────────── Vector 核私有 ────────────┐                      │
   │  UB (Unified Buffer, 逐元素工作台)            │  ← T.alloc_ub       │
   │     ┌──────────┐  ┌──────────┐                 │                     │
   │     │ X_UB     │  │ Y_UB     │                 │                     │
   │     └──────────┘  └──────────┘                 │                     │
   └────────────────────────────────────────────────┘                     │
        ▲                                                                  │
        │ DMA (T.copy GM↔UB)                                               │
        └──────────────────────────────────────────────────────────────────┘
```

> **人话**:GM 是城外大仓,L1 是 Cube 核的小仓,L0C 是 Cube 核的累加器寄存器,UB 是
> Vector 核的工作台。每两层之间都要靠 DMA(T.copy)显式搬运,靠 T.barrier\_all 同步。

### 5. GELU kernel:Vector 核 + UB + buffer 级原语

源码:`examples/tilelang_ascend/src/gelu_tilelang.py`。GELU 是逐元素算子,跑在 **Vector 核**,
和 GEMM(Cube 核)走的是**完全不同的通路**。

#### 5.1 kernel 设计

```python
@tilelang.jit(out_idx=[-1])
def gelu_activation(N: int, BLOCK: int, dtype: str = "float16"):
    num_blocks = N // BLOCK

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks) as cid:
            X_UB = T.alloc_ub((BLOCK,), dtype)   # x (只读)
            T1   = T.alloc_ub((BLOCK,), dtype)   # 中间结果 1
            T2   = T.alloc_ub((BLOCK,), dtype)   # 中间结果 2
            ONES = T.alloc_ub((BLOCK,), dtype)   # 常数 1
            Y_UB = T.alloc_ub((BLOCK,), dtype)   # 最终输出
            start = cid * BLOCK

            T.copy(X[start : start + BLOCK], X_UB)
            T.ascend_tile.fill(ONES, ONE)

            # Vector 流水: 13 条 buffer 级指令, 整 BLOCK 一条 Vector 指令
            T.ascend_tile.mul(T1, X_UB, X_UB)         # (1)  T1   = x * x
            T.ascend_tile.mul(Y_UB, T1, X_UB)         # (2)  Y_UB = x^2 * x = x^3
            T.ascend_tile.mul(T1, Y_UB, CCUB)         # (3)  T1   = CCUB * x^3
            T.ascend_tile.add(Y_UB, X_UB, T1)         # (4)  Y_UB = x + CCUB*x^3
            T.ascend_tile.mul(T1, Y_UB, CSQRT)        # (5)  T1   = CSQRT * (...)
            T.ascend_tile.add(Y_UB, T1, T1)           # (6)  Y_UB = 2*inner
            T.ascend_tile.exp(T1, Y_UB)               # (7)  T1   = exp(2*inner) = e2
            T.ascend_tile.add(Y_UB, T1, ONE)          # (8)  Y_UB = e2 + 1
            T.ascend_tile.sub(T2, T1, ONES)          # (9)  T2   = e2 - 1 (sub 只接 Buffer)
            T.ascend_tile.div(T1, T2, Y_UB)          # (10) T1   = (e2-1)/(e2+1) = tanh(inner)
            T.ascend_tile.add(Y_UB, T1, ONE)          # (11) Y_UB = 1 + tanh(inner)
            T.ascend_tile.mul(T1, Y_UB, HALF)         # (12) T1   = 0.5*(1+tanh)
            T.ascend_tile.mul(Y_UB, T1, X_UB)         # (13) Y_UB = GELU(x)

            T.copy(Y_UB, Y[start : start + BLOCK])
    return main
```

#### 5.2 关键点

- **`T.alloc_ub`** 而不是 `T.alloc_local`:Vector 核 UB 是 scope=shared,允许 global↔shared
  DMA。用 `T.alloc_local`(scope=local)会抛 `Unsupported scope: src=global, dst=local`
  (坑 #TL-2)。

- **`T.ascend_tile.<op>(dst, src1, [src2_or_scalar])`** 是 **整 BLOCK 一条 Vector 指令**
  的 buffer 级原语,不是 element-wise 表达式。

  - 在 `for k in T.serial(BLOCK): Y_UB[k] = T.exp(X_UB[k])` 里写 element-wise,会生成通用
    `tir.exp` Call,不在 CodeGen 白名单里,抛 `Unresolved call Op(tir.exp)`(坑 #TL-3)。

  - 正确姿势:整 BLOCK 一条 `T.ascend_tile.exp(T1, Y_UB)`。

- **`add/mul`** **接受 Python float scalar**(内部走 `ascend_adds / ascend_muls`,
  广播标量);**`sub/div`** **只接受 Buffer**——常量 1 必须先
  `T.ascend_tile.fill(ONES, 1.0)` 填进 ONES buffer 再用向量 sub。

#### 5.3 `T.Kernel` 解包:带不带 `is_npu` 决定写法

```python
# GEMM(多核 Cube 调度): is_npu=True → 返回二元序列, 必须元组解包
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    ...

# GELU/Softmax(本仓库逐元素版): 不带 is_npu → 单变量即可
with T.Kernel(num_blocks) as cid:
    ...
```

> 规则:**`is_npu=True` 时** `T.Kernel` 返回二元序列,只写 `as cid` 会抛解包错误;
> **不带 `is_npu` 时**单变量即可。两种写法在本仓库源码里都真实存在,以源码为准。

#### 5.4 注解 workaround:`#TL-5`

ascend wheel 0.1.1.010 的 parser 要求注解是**实际 Buffer 对象**,不能是字符串。所以:

1. **顶部禁止** `from __future__ import annotations`(否则注解被惰性保留为 str)。
2. 外层闭包参数 `N / BLOCK / dtype` 要在定义 `@T.prim_func` 之前,通过
   `sys.modules[__name__].__dict__[...] = ...` 临时注入模块 globals(`return main` 后
   在 `finally` 里还原)。完整 workaround 见 `gelu_tilelang.py` L50-L115 和
   `softmax_tilelang.py` L50-L115。

### 6. Softmax kernel:`T.serial` 手工 reduction

源码:`examples/tilelang_ascend/src/softmax_tilelang.py`。Softmax 带沿最后一维的 reduction,
tilelang-ascend v0.1.1.010 没有暴露显式 `ReduceMax` / `ReduceSum` 原语,本教学版用 **`T.serial` 循环手工**完成 4 个阶段:

```
y[i] = exp(x[i] - m) / Σ_j exp(x[j] - m),  其中 m = max_j x[j]

Phase 1 : 串行比较, 找出整行最大值 m
Phase 2 : 逐元素 exp(x[i] - m)
Phase 3 : 串行累加, 求出分母 sum_exp
Phase 4 : 逐元素 exp_val / sum_exp (广播除法)
```

#### 6.1 kernel 主体

```python
@T.prim_func
def main(X: "T.Tensor((D,), dtype)", Y: "T.Tensor((D,), dtype)"):
    with T.Kernel(num_blocks) as cid:
        X_UB = T.alloc_local((BLOCK,), dtype)
        Y_UB = T.alloc_local((BLOCK,), dtype)
        M_UB = T.alloc_local((1,), dtype)    # 整行最大值 m
        S_UB = T.alloc_local((1,), dtype)    # Σ exp(x - m)
        INV_UB = T.alloc_local((1,), dtype)   # 1 / S_UB

        start = cid * BLOCK
        T.copy(X[start : start + BLOCK], X_UB)

        # Phase 1: 串行求整行最大值
        M_UB[0] = X_UB[0]
        for k in T.serial(1, BLOCK):
            xv = X_UB[k]
            if xv > M_UB[0]:
                M_UB[0] = xv

        # Phase 2: 逐元素 exp(x - m), 写入 Y_UB 暂存
        for k in T.serial(BLOCK):
            diff = X_UB[k] - M_UB[0]
            Y_UB[k] = T.exp(diff)

        # Phase 3: 串行求和 S = Σ Y_UB[k]
        S_UB[0] = Y_UB[0]
        for k in T.serial(1, BLOCK):
            S_UB[0] = S_UB[0] + Y_UB[k]

        # Phase 4: inv = 1/S, 逐元素 Y = exp_val * inv
        INV_UB[0] = 1.0 / S_UB[0]
        for k in T.serial(BLOCK):
            Y_UB[k] = Y_UB[k] * INV_UB[0]

        T.copy(Y_UB, Y[start : start + BLOCK])
```

#### 6.2 关键点

> ⚠️ **待迁移旧写法**:本节代码忠实镜像仓库现状——`softmax_tilelang.py` 仍用
> `T.alloc_local` 接 `T.copy(GM→X_UB)`(上框 L711)。但同样的组合在 GELU 上对
> v0.1.1.010 实测会抛 `Unsupported scope: src=global, dst=local`(坑 #TL-2,见
> `gelu_tilelang.py` 注释)。**如果你跑 softmax 报这个错,先把 `T.alloc_local` 改成
> `T.alloc_ub`**(对照上文 §5.1 GELU 写法)。

- **`T.alloc_local`** 的定位:存标量 `M_UB / S_UB / INV_UB`(各 1 个元素)这类
  thread-private 中间值;**接 `T.copy(GM, ...)` 的缓冲必须用 `T.alloc_ub`**(坑 #TL-2)。

- **`T.serial`** **做 reduction**:因为 tilelang-ascend v0.1.1.010 没有暴露显式 ReduceMax/ReduceSum,本教学版
  用 `for k in T.serial(...)` + 标量累加器手工完成。性能不极致,但**语义最清晰**,
  适合理解 Ascend Vector 核的 reduction 怎么写。

- **字符串前向引用**:`main(X: "T.Tensor((D,), dtype)", ...)` —— Softmax 这里**开了**
  `from __future__ import annotations`,把注解写成字符串,再通过模块 globals 注入绕过
  typing 求值。和 GELU 的"不开 future + 裸注解 + globals 注入"是**两种不同的 workaround**,
  二者都能绕过 #TL-5(详见该坑说明)。

### 7. 正确性验证:test\_gemm.py

源码:`examples/tilelang_ascend/src/test_gemm.py`。流程:

1. **设置** **`ACL_OP_INIT_MODE=1`**(在 `import torch_npu` 之前)——跳过 torch\_npu 的
   TBE/GE 算子编译器初始化,避免和 tilelang 自带 TVM 的 FFI 冲突。
2. 生成 fp16 随机矩阵 A, B(torch tensor,放到 npu 设备)。
3. **预热**:首次 `gemm(a, b)` 触发 tilelang-ascend 编译,这一步慢(TVM 编译 + CANN
   链接,见 [FAQ Q4](#q4-首次编译慢-正常吗))。
4. 正式计时:`gemm(a, b, block_M=128, block_N=128, K_L1=64)`。
5. numpy 算参考(fp16 输入升 fp32 累加)。
6. `np.allclose(c_np, c_ref, atol=1e-2, rtol=1e-2)` 校验(fp16 容差)。

```python
import os
os.environ.setdefault("ACL_OP_INIT_MODE", "1")   # 关键:在 import torch_npu 之前

import tilelang                                  # 触发后端注册
import torch
import torch_npu                                 # 注册 npu 设备

from gemm_tilelang import gemm

M = N = K = 128
a_np = np.random.randn(M, K).astype(np.float16)
b_np = np.random.randn(K, N).astype(np.float16)
a = torch.from_numpy(a_np).to("npu")
b = torch.from_numpy(b_np).to("npu")

c_ref = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)

_ = gemm(a, b)                                   # 预热:首次编译

start = time.perf_counter()
c = gemm(a, b, block_M=128, block_N=128, K_L1=64)
elapsed = time.perf_counter() - start

c_np = c.cpu().numpy()
ok = bool(np.allclose(c_np, c_ref, atol=1e-2, rtol=1e-2))
print(f"耗时: {elapsed*1000:.4f} ms, max_abs_error=..., {ok=}")
```

预期输出:

```
[INFO] === TileLang-Ascend GEMM 测试 (dtype=float16) ===
[INFO] 预热编译 (首次调用触发 tilelang-ascend 编译)...
[INFO] TileLang kernel 耗时: 0.3788 ms
[INFO] 校验结果: PASS (max_abs_error=9.765625e-04, atol=1e-2, rtol=1e-2)
[INFO] TileLang-Ascend GEMM 测试完成, 全部 PASS
```

GELU / Softmax 的验证脚本同样在仓库里（TileLang 后端运行时曾受容器 HDC 链路问题影响，
见 [ops/05 §8.10](/ops/05-gelu) 的验证步骤与排障）：

```bash
cd examples/tilelang_ascend
uv run python src/test_gelu.py       # GELU 正确性 (max_err < 5e-3)
uv run python src/test_softmax.py    # Softmax 正确性
```
### 8. 性能特征

#### 8.1 128³ GEMM 实测对比

| DSL                            | NPU run | max\_abs\_error | 耗时          | 状态       |
| ------------------------------ | ------- | --------------- | ----------- | -------- |
| Python(CPU 基准)                 | —       | 0.0             | 4.27 s      | PASS     |
| Triton (triton-ascend)         | ✅       | 0.0             | 0.79 ms     | PASS     |
| **TileLang (tilelang-ascend)** | ✅       | **9.77e-04**    | **0.38 ms** | **PASS** |
| Ascend C                       | ✅       | 0.0             | —           | PASS     |

> **人话**:同是 128³ fp16 GEMM,TileLang 0.38 ms 是该规模四种 DSL 里最快的写法,
> 比 Triton 0.79 ms 快一倍,比 Python 朴素三重循环快一万倍以上。

#### 8.2 显式调度效率:为什么 TileLang 在这个规模下最快

差距不来自数学公式,而来自**显式调度**带来的几个效率点:

- **L1 / K\_L1 显式分配**:`T.alloc_L1((block_M, K_L1), ...)` 明确告诉后端"这份数据进 L1,
  L1 的容量规划是 block\_M × K\_L1"。后端不用猜搬运粒度,可以直接生成最高效的 DMA。

- **双缓冲潜力**:`T.set_flag` / `T.wait_flag` 可以手动多级流水(本教学版没用,
  官方 `example_gemm_intrinsic.py` 有),让 MTE2(搬)和 Cube(算)充分并行。

- **`init=(k==0)`** **直接交给 Cube 单元**:不写显式 clear,首块清零 + 后续累加一条 Cube
  指令完成,语义最贴近硬件。

- **`T.barrier_all`** **只在必要处同步**:不是无脑 full barrier,后端 lower 时会按队列依赖
  生成最小同步。

#### 8.3 块参数选择:`block_M=128, block_N=128, K_L1=64`

| 参数 | 取值 | 选择理由 |
| --- | --- | --- |
| `block_M` | 128 | Cube 按 16×16 MAC 阵列做矩阵乘,128 是 16 的倍数;同时让 A_L1(128 × 64 fp16 = 16 KB)和 C_L0(128 × 128 fp32 = 64 KB)合理占用 L1/L0C。 |
| `block_N` | 128 | 同上,让 B_L1(64 × 128 fp16 = 16 KB)合理占用 L1。 |
| `K_L1` | 64 | 一次搬 64 个 K 到 L1。K=128 时分 2 次搬,刚好展示 `init` 的"首块清零 + 后续累加"语义(教学版有意选 64 而非 128,让 K 维分块可见)。 |
| `dtype` | fp16 | Cube 原生精度,带宽最省。 |
| `accum_dtype` | fp32 | K 维累加用 fp32,避免 fp16 累加溢出(混合精度标准做法)。 |

> **人话**:128 / 128 / 64 不是玄学,是**让 L1 和 L0C 都装得下 + 让 K 维有可观察的分块**。
> 真正生产环境会做 sweep 找最优,本教学版的取值兼顾了"够快 + 够好懂"。

### 9. GEMM 调度流程图(Mermaid)

```mermaid
sequenceDiagram
    autonumber
    participant Host as Host (Python)
    participant GM as GM (HBM)
    participant MTE2 as MTE2 队列 (GM→L1 搬运)
    participant L1 as L1 缓冲
    participant MTE1 as MTE1 队列 (L1→L0A/B)
    participant L0AB as L0A / L0B
    participant Cube as Cube 单元
    participant L0C as L0C 累加器
    participant MTE3 as MTE3 队列 (L0C→GM 写回)

    Host->>MTE2: T.copy(A[bx*block_M, k*K_L1], A_L1)
    Host->>MTE2: T.copy(B[k*K_L1, by*block_N], B_L1)
    Note over Host,L1: T.barrier_all() — 等 MTE2 搬完
    MTE2->>L1: A_L1 / B_L1 就位
    L1->>MTE1: MTE1 队列搬运 L1→L0A/L0B
    Note over Host,Cube: (TileLang 把 MTE1 + Cube 视作整体, 不需用户显式同步)
    MTE1->>L0AB: A_L0A / B_L0B 就位
    Cube->>L0C: T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))
    Note over Host,L0C: T.barrier_all() — 等 Cube 算完
    Note over Host: for k in T.serial(loop_k) 循环到下一块
    L0C->>MTE3: 所有 K 块累加完, T.copy(C_L0, C[...])
    MTE3->>GM: 写回 fp16 输出块
```

> **人话**:一个输出块的生命周期是 **搬 A/B 到 L1 → 同步 → Cube 算进 L0C → 同步 →(循环 K)→ 累加完 → L0C 写回 GM**。每一步都是显式的,你写哪步就有哪步。

### 10. 显式内存层级映射图(Mermaid)

```mermaid
flowchart TB
    subgraph GM_LAYER["GM (HBM) 层"]
        AG["A: (M, K) fp16"]
        BG["B: (K, N) fp16"]
        CG["C: (M, N) fp16"]
    end

    subgraph CUBE_CORE["Cube 核"]
        direction TB
        L1A["L1: A_L1\n(block_M, K_L1) fp16\nT.alloc_L1"]
        L1B["L1: B_L1\n(K_L1, block_N) fp16\nT.alloc_L1"]
        L0A["L0A: Cube A 输入\n(自动)"]
        L0B["L0B: Cube B 输入\n(自动)"]
        L0C["L0C: C_L0\n(block_M, block_N) fp32\nT.alloc_L0C"]
        L1A -->|MTE1| L0A
        L1B -->|MTE1| L0B
        L0A --> Cube["Cube 单元\n16×16 MAC 阵列"]
        L0B --> Cube
        Cube --> L0C
    end

    subgraph VEC_CORE["Vector 核"]
        UB["UB: X_UB / Y_UB\n(BLOCK,) fp16\nT.alloc_ub"]
    end

    AG -->|T.copy GM→L1\nDMA| L1A
    BG -->|T.copy GM→L1\nDMA| L1B
    L0C -->|T.copy L0C→GM\nDMA + fp32→fp16| CG
    AG -.->|T.copy GM→UB\nDMA| UB
    UB -.->|T.copy UB→GM\nDMA| CG
```

> **人话**:TileLang 把 GM(城外大仓)、L1(Cube 小仓)、L0C(Cube 累加器)、UB(Vector
> 工作台)显式暴露,你写哪条 `T.copy`,DMA 就走哪条路。每一步都看得见。

***

## 图表汇总

本文共使用 4 张 Mermaid 架构图 + 2 张 ASCII 心法图，汇总如下：

| # | 图表                            | 位置        | 说明                                                        |
| - | ----------------------------- | --------- | --------------------------------------------------------- |
| 1 | Mermaid：编译通路                  | §1 工具链    | @tilelang.jit → TileLang IR → ascend 后端 → Ascend C → CANN |
| 2 | Mermaid：抽象梯子                  | §2 Why    | Python → Triton → TileLang → Ascend C 控制力递增               |
| 3 | Mermaid：GEMM 调度时序             | §9 调度流程 | GM→L1→L0C→Cube→L0C→GM 的完整 sequenceDiagram                 |
| 4 | Mermaid：显式内存层级映射              | §10 层级映射 | GM/HBM→L1→L0A/L0B→L0C→UB 的数据流 flowchart                   |
| 5 | ASCII：TileLang vs Triton 调度对比 | §4.9      | 显式 T.copy/T.barrier vs 隐式 tl.load/tl.store                |
| 6 | ASCII：L1/L0C/UB 内存层级          | §4.10     | 城外大仓→Cube 小仓→Cube 累加器→Vector 工作台                          |

> **人话**：4 张 Mermaid 画的是编译通路和调度时序，2 张 ASCII 画的是心智模型——
> 一张对比 Triton 隐式 vs TileLang 显式，一张画硬件层级。

***

## FAQ

### Q1. `is_npu=True` 不加会怎样?

`T.Kernel(grid)` **不带** `is_npu=True`,tilelang 会按 **GPU CUDA thread block** 语义生成
IR(用 `threadIdx` / `blockIdx`)。加载时:

- 要么报 `target ascend_npu not found`(因为根本没走 ascend 后端)。

- 要么报 `Cannot find global function cce.product_init`(TVM FFI 注册表冲突)。

**正确写法**:`T.Kernel(grid, is_npu=True) as (cid, _)`(解包成二元序列)。

### Q2. `K_L1` 怎么选?

`K_L1` 是 K 维一次搬到 L1 的粒度。选值需要平衡:

- **太大**:A\_L1 + B\_L1 占满 L1,挤掉 C\_L0 的空间,或者超出 L1 容量。

- **太小**:DMA 次数多,MTE2 队列开销占比上升,带宽利用率下降。

- **经验**:让 A\_L1 + B\_L1 + C\_L0 总占用落在 L1 + L0C 容量内(910B2 上 L1 ≈ 1MB、L0C
  ≈ 256 KB),`K_L1=64` 是 128³ fp16 场景下一个稳妥的起点(教学版有意取 64 让 K 维分块
  可见)。生产环境做 sweep 找最优。

### Q3. `T.gemm_v0` 的 `init` 参数语义是什么?

- `init=True` → `C_L0 = A_L1 @ B_L1`(清零累加器再乘)。

- `init=False` → `C_L0 += A_L1 @ B_L1`(累加)。

- `init=(k == 0)` → **首块清零、后续累加**,一行表达完整 K 维累加语义。

这是 Ascend Cube 单元的标准累加语义,避免显式 clear,也避免"首块读到脏 L0C"的 bug。

### Q4. 首次编译慢,正常吗?

正常。TileLang 基于 TVM,首次 `kernel = gemm_matmul(...)` 触发:

1. TileLang 前端解析 → TileLang IR。
2. tilelang-ascend 后端 lower → AscendNPU IR / Ascend C。
3. CANN 工具链链接 → `.o` / kernel module。

整个过程几十秒到数分钟,后续相同参数走 `~/.tilelang/cache` 缓存,毫秒级加载。

### Q5. `T.barrier_all` 的作用是什么?

Ascend 的 MTE2(GM→L1 搬运队列)、MTE1(L1→L0A/L0B 搬运队列)、MTE3(L0C→GM 写回队列)
和 Vector 队列彼此**异步**。`T.barrier_all()` 是一个**全队列同步 barrier**:

- 保证"GM→L1 搬完再 L1→L0A/L0B"(MTE2 → MTE1)。

- 保证"L0C 算完再写回 GM"(MTE3 → 后续 MTE2)。

- 保证"下一轮 K 循环开始前,A\_L1/B\_L1 已被消费完"(避免覆写未读数据)。

不加 `T.barrier_all` 会导致数据竞争(读到未搬完的数据,或覆写未读完的 buffer)。

### Q6. tilelang 和 tilelang-ascend 的版本兼容怎么对应?

- PyPI 上的 `tilelang`(如 0.1.13)是 **CUDA 版,不含 ascend 后端**。

- tilelang-ascend 的预编译 wheel 以同名 `tilelang` 包发布,文件名带 `cannXXX` 后缀
  (如 `tilelang-0.1.1.10+ubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl`)。

- 必须选匹配 **CANN 版本 + 架构 + Python 版本** 的 wheel。装错版本会报
  `target ascend_npu not found` 或 `undefined symbol ... IRBuilderFrameNode`。

- 本仓库锁 `tilelang-ascend 0.1.1.010`(cann900 + aarch64 + cp311),和 torch 2.8.0 +
  torch\_npu 2.8.0rc1 共用 venv。

### Q7. `T.alloc_ub` vs `T.alloc_local` 怎么选?

- **`T.alloc_ub`** → scope=shared → 映射到 NPU Vector 核的 **UB**,允许 global↔shared DMA。
  凡是要接 `T.copy(GM, ...)` 的 Vector 缓冲,**必须**用 `alloc_ub`。

- **`T.alloc_local`** → scope=local → thread-private 本地缓冲。**不能**接
  `T.copy(GM, local)`,会抛 `Unsupported scope: src=global, dst=local`(坑 #TL-2)。
  仅在"标量 / 纯内部计算、不接 GM DMA"的场景用(如 Softmax 的 M\_UB/S\_UB/INV\_UB)。
  注意:仓库里 `softmax_tilelang.py` 的 X\_UB/Y\_UB 仍是 `alloc_local` 接 GM 的旧写法,
  报 Unsupported scope 时先改成 `alloc_ub`(见 §6.2 的待迁移警告)。

### Q8. `Cannot find global function cce.product_init` 怎么办?

这是 tilelang 自带的 TVM 与 CANN 的 `te` 模块共享 TVM FFI 全局注册表、互相覆盖导致。解决:

```python
import os
os.environ.setdefault("ACL_OP_INIT_MODE", "1")  # 必须在 import torch_npu 之前
```

跳过 torch\_npu 的 TBE/GE 算子编译器初始化。本测试只做张量分配 + tilelang 自管 kernel
launch,不走 torch\_npu 图编译,故可安全跳过。详见 `test_gemm.py` 顶部注释。

***

## TL;DR 末尾汇总

1. **TileLang** 是北大杨智团队开源的 tiled kernel DSL,基于 TVM;**tilelang-ascend** 是它对接
   Ascend NPU 的后端,把 TileLang IR 编译成 AscendNPU IR / Ascend C → CANN 可执行 kernel。
2. 在四种 DSL 抽象梯子里,TileLang 位于 **Triton 之下、Ascend C 之上**:比 Triton 多了
   "显式指定 L1/L0C + Cube 调用"的控制力,比 Ascend C 少一层"裸 C++ + bisheng 直编"
   的复杂度。
3. **核心原语**:`@tilelang.jit` / `T.prim_func` / `T.Kernel(is_npu=True)` /
   `T.alloc_L1` / `T.alloc_L0C` / `T.alloc_ub` / `T.Scope("C"/"V")` / `T.copy` /
   `T.gemm_v0(init=...)` / `T.barrier_all` / `T.serial` / `T.ceildiv` /
   `T.ascend_tile.<op>`。
4. **GEMM 调度链**:GM → L1(DMA)→ 同步 → Cube 算进 L0C → 同步 →(循环 K)→ L0C 写回 GM。
   每一步都**显式写在代码里**,编译器不替你决定。
5. **实测**:128³ fp16 GEMM **0.38 ms**(Ascend 910B2 + CANN 9.0.0),9.77e-04 误差,全 PASS。
   比同规模 Triton 0.79 ms 快一倍。
6. **坑速查**:`is_npu=True` 不能漏、`@T.prim_func` 注解别开 future annotations(坑 #TL-5)、
   Vector UB 用 `alloc_ub` 不要用 `alloc_local`(坑 #TL-2)、Vector buffer 级原语走
   `T.ascend_tile.<op>` 不要写 element-wise(坑 #TL-3)、`import torch_npu` 前设
   `ACL_OP_INIT_MODE=1`(坑 #TL-1)。

***

## 参考资料

**官方 / 项目来源:**

- TileLang GitHub: [https://github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang)

- tilelang-ascend GitHub: [https://github.com/tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)

- tilelang-ascend releases(预编译 wheel): [https://github.com/tile-ai/tilelang-ascend/releases](https://github.com/tile-ai/tilelang-ascend/releases)

- 华为昇腾 CANN 文档中心: [https://www.hiascend.cn/document](https://www.hiascend.cn/document)

- 华为昇腾 Ascend C 官方: [https://www.hiascend.com/cann/ascend-c](https://www.hiascend.com/cann/ascend-c)

**本仓库文件引用(可本地核验):**

- `examples/tilelang_ascend/README.md` — 工具链安装 / 编译通路 / 常见问题总览

- `examples/tilelang_ascend/src/gemm_tilelang.py` — GEMM kernel + `gemm()` 封装

- `examples/tilelang_ascend/src/gelu_tilelang.py` — GELU Vector 核 kernel(13 条
  `T.ascend_tile` 指令)

- `examples/tilelang_ascend/src/softmax_tilelang.py` — Softmax 4 阶段手工 reduction

- `examples/tilelang_ascend/src/test_gemm.py` — 正确性验证(`ACL_OP_INIT_MODE=1` +
  预热 + allclose)

- `examples/tilelang_ascend/pyproject.toml` — uv 配置(numpy + torch==2.8.0)

- `docs/pages/dsl/00-dsl-overview.md` — 四种 DSL 总览(抽象梯子 + 实测对比)

- `docs/pages/hardware/02-storage-hierarchy.md` — 存储层级(GM/L1/L0A/L0B/L0C/UB/DMA)

- ops/05-gelu §8.10.4 — TileLang-Ascend 五大常见坑(#TL-1..#TL-5)

- `docs/pages/ops/09-gemm.md` — GEMM 算子级原理(Cube 16×16 MAC、混合精度、tiling)

