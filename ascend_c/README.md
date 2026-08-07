# ascend_c/ — Ascend C kernel DSL

本目录用 **Ascend C**(CANN 原生 C++ kernel DSL)实现一个朴素 GEMM kernel,
并在 host 侧用 ACL 驱动它运行、做正确性校验。

## Ascend C 是什么

Ascend C 是华为 CANN 提供的**底层 kernel 编程 DSL**:用 C++ 写算子,
经 `bisheng`(毕昇)编译器编成 AI Core 可执行的机器码。它直接暴露硬件资源
(片上内存、Cube/Vector 单元、多核),性能上限最高,但开发门槛也最高。

> 与 TBE/AscendC 的关系:Ascend C 是较新的统一 kernel 编程模型,
> 取代了早期的 TBE(Tensor Boost Engine)写法。

## 工具链

```
gemm_kernel.cpp  ──[bisheng 编译器]──►  gemm_kernel.o  (device 机器码)
                                                │
gemm_host.cpp  ──[g++ + ACL]──►  ascend_gemm  (host 可执行)
                                                │
                        运行: aclrtLaunchKernel("gemm_kernel", ...)
                                                │
                        NPU AI Core 执行 ──►  结果回传 host 校验
```

- **bisheng(毕昇)编译器**:CANN 的 ASC 编译器前端,把 C++ kernel 源码编成 device `.o`。
  在本机(CANN 9.0.0, aarch64)由 `$ASCEND_HOME_PATH/bin/bisheng` 提供(source `set_env.sh` 后在 PATH 中)。
  > 注:老文档里常写的 `ascendc` 可执行在本版本里**不存在**——`$ASCEND_HOME/compiler/ascendc` 只是个含 `include/` 的目录;
  > 官方 Ascend-C 构建框架(`ascendc_kernel_cmake`)内部 `find_program(... NAMES bisheng ...)`,用的也是 bisheng。
- **ACL**(Ascend Computing Language):host 侧运行时库,提供 `aclrtMalloc`/`aclrtMemcpy`/`aclrtLaunchKernel`
  等接口(类比 CUDA Runtime 的 `cudaMalloc`/`cudaMemcpy`/`cudaLaunchKernel`)。
- **CANN**:整套软件栈,提供头文件 `acl/acl.h` 与库 `libascendcl.so`/`libacl.so`。

### 构建状态(2026-08-07 实测)

- 远程 NPU:`npu-smi` 显示 **910B2**,CANN 9.0.0,`bisheng` 可用。
- 本目录的 **kernel 源码 + host 源码 + 注释已完整**,可直接学习 Ascend C 的写法。
- **device 构建待最终调通**:`bisheng` 直接调用需要 `--cce-soc-version` + `--cce-soc-core-type` 组合;
  实测 910B1/B2 上,常见的 core_type 值(`AiCore`/`AICore`/`MIX`/`VectorCore` 等)均被拒
  (报 `soc_core_type: X is not supported for soc_version Ascend910B1`)。
  CANN 自带的 `SOC_MAP_EXT` 把 `ascend910b` 家族映射到 `soc_version=Ascend910B1`。
- 正确的标志组合需走官方 `ascendc.cmake` 框架(它会按 target 自动构造 soc 标志)。
  本目录 CMakeLists 已把 `ASCEND_SOC_VERSION` / `ASCEND_SOC_CORE_TYPE` 暴露为 cache 变量,
  便于远程用 `-DASCEND_SOC_CORE_TYPE=...` 调参;若直调仍不通,改用官方 op-project 框架构建(见下节"官方框架路径")。

## NPU 硬件概念(理解 kernel 注释所需)

| 概念 | 说明 |
|---|---|
| **AI Core** | NPU 的计算核心,多个核可并行(本朴素版只用 1 个) |
| **Cube 单元** | 专门做矩阵乘(16×16 粒度)的硬件单元,fp16 原生精度 |
| **Vector 单元** | 向量运算单元 |
| **GM (Global Memory)** | HBM 显存,容量大但带宽低,kernel 通过 `GlobalTensor` 访问 |
| **L1 / UB (Unified Buffer)** | 片上高速缓存,需显式搬运(CopyData)才能复用数据 |
| **tiling** | 把大矩阵切成能装进 UB 的小块,是性能优化的核心 |

> 本朴素 kernel **不用** Cube / UB / tiling,直接逐元素从 GM 读写 —— 正确但极慢,
> 注释中标注了优化方向。这是理解"为什么要优化"的起点。

## 精度策略

- 输入/输出:`half`(float16)—— Cube 单元原生精度,吞吐最高。
- 累加器:`float`(fp32)—— 避免 fp16 在大 K 下累加溢出(混合精度)。
- 校验容差:`atol=1e-2`。

## 如何构建与运行

```bash
# 1. 加载 CANN 环境 (必须, 否则 ASCEND_HOME_PATH / bisheng 找不到)
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. 构建 (host 可执行 + 尝试编 kernel.o)
cd ascend_c
cmake -S . -B build
cmake --build build

# 3. 运行 (需要 kernel.o 已成功编出)
./build/ascend_gemm
```

若 `bisheng` 编 kernel 时报 `soc_core_type ... is not supported`,可用 cache 变量调参:
```bash
cmake -S . -B build -DASCEND_SOC_VERSION=Ascend910B1 -DASCEND_SOC_CORE_TYPE=<待确认值>
```

预期输出(kernel 编出后):
```
ascend_c GEMM: PASS (max_abs_error=0, M=N=K=128, dtype=fp16)
```

### 官方框架路径(若直调 bisheng 不通)

CANN 自带 Ascend-C op-project 框架,路径:
`$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/`,入口 `ascendc.cmake`。
它内部用 `find_program(NAMES bisheng)` + 按 target 自动构造 soc 标志,是官方推荐写法。
模板在 `$ASCEND_HOME_PATH/tools/op_project_templates/ascendc/`。
本目录为**教学简化版**(直接 bisheng + 自写 ACL host),突出 kernel 写法本身;
若需移植到官方框架,按上述模板的 `op_kernel/` + `op_host/` + `op_proto/` 结构组织即可。

## 文件说明

| 文件 | 作用 |
|---|---|
| `op_kernel/gemm_kernel.cpp` | Ascend C kernel:朴素三重循环,fp16+fp32 累加,详细注释 |
| `src/gemm_host.cpp` | host 驱动:ACL 初始化、H2D/D2H、kernel 启动、CPU 参考+校验 |
| `CMakeLists.txt` | 构建:find bisheng → 编 kernel.o,g++ 编 host;soc 标志可调 |
| `BUILDING.md` / `HACKING.md` | 通用 CMake 说明(非 Ascend 专属) |

## 常见问题

- **`ASCEND_HOME_PATH 未设置`**:没 source `set_env.sh`,或在本 shell 重新 source。
- **`bisheng: error: Unsupported CCE architecture ... soc_core_type`**:910B 上 core_type 标志组合待确认,见上方"构建状态";先用 `-DASCEND_SOC_CORE_TYPE=` 调参,或走官方 `ascendc.cmake` 框架。
- **`fatal error: 'acl/acl.h' file not found`**:CANN include 路径未配置,确认 `set_env.sh` 已 source。
- **精度误差大**:确认累加器是 `float` 而非 `half`;朴素版逐元素读 GM 慢属正常现象。
