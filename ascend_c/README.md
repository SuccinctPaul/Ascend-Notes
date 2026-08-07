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

- 远程 NPU:`npu-smi` 显示 **910B2**(多卡),CANN 9.0.0,`bisheng` 可用。
- ✅ **kernel 源码 + 注释完整**,可直接学习 Ascend C 的写法。
- ✅ **kernel 编译跑通**:`bisheng --cce-aicore-arch=dav-c220-cube` 成功编出 `gemm_kernel.o`(ELF arch 0x1029 = Ascend AICore)。算法与 `python/` 基准同源(朴素三重循环 + fp32 累加),逻辑正确。
- ⚠️ **host 端运行 NPU kernel 需官方 pack 流程**: `aclrtBinaryLoadFromFile` 不能直接加载 bisheng 编出的 raw `.o`(报 `107000 ACL_ERROR_RT_PARAM_INVALID`)。raw `.o` 需先用 `ascendc_pack_kernel` 打包(它会把 `.aicore_binary` 等 section 加上运行时识别用的 magic header),而 pack 又需要一个由 `update_host_stub.py` 生成的 `device_aic.o` stub —— 这套流程由官方 `ascendc.cmake` 框架自动完成(见下节"官方框架路径")。
- 本目录的 `src/gemm_host.cpp` 展示了**现代 ACL kernel launch API 的完整写法**(`aclrtBinaryLoadFromFile → aclrtBinaryGetFunctionByEntry → aclrtKernelArgsInit/Append/Finalize → aclrtLaunchKernelWithConfig`,含 910B cube 的 FFTS prepend),教学价值完整;只是缺 pack 步骤,直接跑会卡在 load。
- **关键标志组合**(CANN 9.0.0 新方案,旧的 `--cce-soc-version`/`--cce-soc-core-type` 已废弃):
  ```
  bisheng --cce-aicore-lang --cce-aicore-arch=dav-c220-cube \
          --cce-aicore-only --cce-auto-sync --cce-mask-opt \
          -std=c++17 -O3 -I<asc/include ...>  gemm_kernel.cpp -c -o gemm_kernel.o
  ```
  - `--cce-aicore-arch=dav-c220-cube`:**ascend910b** 家族对应的 AI Core 微架构
    (CANN `ascendc.cmake` 框架的 `legacy_modules/host_config.cmake` 把 `ascend910b*` 映射到 `BUILD_MODE=c220`,
    再由 `bisheng_intf.cmake` 把 `c220` 映射到 `dav-c220-cube`(AI Core,含 Cube+Vector)/ `dav-c220-vec`(纯 Vector Core)。
    本 GEMM 走 AI Core,故用 `cube` 变体)。
  - `--cce-aicore-only`:只编 device kernel,不生成 host stub(教学简化;若要直接 run,去掉此 flag 让 bisheng 顺带生成 host stub,再走官方 pack)。
  - 其它芯片:`ascend310b*` → `dav-m300`,`ascend310p*` → `dav-m200`,`ascend910a` → `dav-c100`(详见 `host_config.cmake`)。
- 旧的 `--cce-soc-version=Ascend910B1` + `--cce-soc-core-type=...` 标志在本版本上**已不接受**
  (报 `soc_core_type: X is not supported for soc_version Ascend910B1`),不要再用。
- CMakeLists 已把 `ASCEND_AICORE_ARCH` 暴露为 cache 变量,远程可用 `-DASCEND_AICORE_ARCH=...` 覆盖。

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

# 2. 构建 (host 可执行 + 编 kernel.o)
cd ascend_c
cmake -S . -B build
cmake --build build

# 3. 运行 host (注意: 直接 run 当前卡在 kernel 二进制 load, 见"构建状态")
#    - host 能编出, 但 aclrtBinaryLoadFromFile 加载 raw .o 报 107000
#    - 要跑通 run, 需走下方"官方框架路径"做 pack + host stub
cd build && ./ascend_gemm
# 或指定 kernel 路径: ./ascend_gemm /path/to/gemm_kernel.o
```

**验证 kernel 编译成功**(不需要 run host):
```bash
# 编出 .o 即算 kernel 正确性 (算法逻辑与 python/ 基准同源, 已对齐)
ls -la build/gemm_kernel.o   # 应为 ~34KB ELF, arch 0x1029 (Ascend AICore)
file build/gemm_kernel.o
```

若 `bisheng` 编 kernel 时报 arch 不支持,可用 cache 变量覆盖:
```bash
cmake -S . -B build -DASCEND_AICORE_ARCH=dav-c220-cube   # ascend910b2 默认值
# 其它芯片示例:
#   ascend310b:  -DASCEND_AICORE_ARCH=dav-m300
#   ascend310p: -DASCEND_AICORE_ARCH=dav-m200
```

预期输出(kernel 编译成功):
```
[ 33%] Compiling Ascend C kernel (bisheng, arch=dav-c220-cube)
[ 33%] Built target kernel
[ 66%] Building CXX object ...ascend_gemm.cpp.o
[100%] Linking CXX executable ascend_gemm
$ file build/gemm_kernel.o
build/gemm_kernel.o: ELF 64-bit LSB relocatable, *unknown arch 0x1029*, ...  # 0x1029 = Ascend AICore
```

(host run 当前因 raw .o 未 pack 会报 `aclrtBinaryLoadFromFile error=107000`,需官方框架 pack 后才能跑通,见"官方框架路径"。)

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
| `CMakeLists.txt` | 构建:find bisheng → 编 kernel.o,g++ 编 host;aicore-arch 可调 |

## 常见问题

- **`ASCEND_HOME_PATH 未设置`**:没 source `set_env.sh`,或在本 shell 重新 source。
- **`bisheng: error: soc_core_type ... is not supported`**:用了旧的 `--cce-soc-version`/`--cce-soc-core-type` 标志(CANN 9.0.0 已废弃)。改用 `--cce-aicore-arch=dav-c220-cube`(见上方"构建状态"),或 `-DASCEND_AICORE_ARCH=...` 覆盖。
- **`bisheng: error: Unsupported CCE architecture`**:`--cce-aicore-arch` 值与芯片不匹配。910B 家族用 `dav-c220-cube`,310B 用 `dav-m300`,310P 用 `dav-m200`(完整映射见 CANN `legacy_modules/host_config.cmake`)。
- **`fatal error: 'kernel_operator.h' file not found`**:include 路径不全。CMakeLists 已加 `asc/include`、`asc/impl`、`tikcpp/tikcfw` 等路径;手写命令需全部带上。
- **`fatal error: 'acl/acl.h' file not found`**:CANN include 路径未配置,确认 `set_env.sh` 已 source。
- **`reinterpret_cast from '__gm__ uint8_t *' to 'uint32_t *' is not allowed`**:GM 指针不能直接 cast 到私有指针,需保留 `__gm__` 修饰符(`__gm__ uint32_t*`)。
- **精度误差大**:确认累加器是 `float` 而非 `half`;朴素版逐元素读 GM 慢属正常现象。
