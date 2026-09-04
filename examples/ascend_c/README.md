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
gemm_kernel.cpp  ──[ascendc.cmake 框架]──►  libgemm.a  (含 device 二进制 + host launch stub)
       │                    │                         │
       │ bisheng 编译       │ ascendc_pack_kernel     │ g++ 链接
       │ device kernel      │ 打包                    │
       ▼                    ▼                         ▼
   device_aiv.o  ──────────►  host_stub.o  ◄──── gemm_host.cpp
                                                        │
                                          ascend_gemm (host 可执行)
                                                        │
                                  运行: aclrtlaunch_gemm_kernel(1, stream, ...)
                                                        │
                                  NPU AI Core 执行 ──►  结果回传 host 校验
```

- **ascendc.cmake 框架**:CANN 官方 Ascend-C 构建系统,路径
  `$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake`。
  提供 `ascendc_library()` 函数,自动完成 bisheng 编译 → host stub 生成 → 打包全流程。
- **bisheng(毕昇)编译器**:CANN 的 ASC 编译器前端,把 C++ kernel 源码编成 device `.o`。
  在本机(CANN 9.0.0, aarch64)由 `$ASCEND_HOME_PATH/bin/bisheng` 提供(source `set_env.sh` 后在 PATH 中)。
- **ACL**(Ascend Computing Language):host 侧运行时库,提供 `aclrtMalloc`/`aclrtMemcpy` 等
  接口(类比 CUDA Runtime 的 `cudaMalloc`/`cudaMemcpy`)。
- **CANN**:整套软件栈,提供头文件 `acl/acl.h` 与库 `libascendcl.so`。

### 构建状态(2026-08-07 实测)

- 远程 NPU:`npu-smi` 显示 **910B2**(多卡),CANN 9.0.0,`bisheng` 可用。
- ✅ **kernel 源码 + 注释完整**,可直接学习 Ascend C 的写法。
- ✅ **kernel 编译 + 打包 + host 运行全部跑通**:用官方 `ascendc.cmake` 框架 (`ascendc_library STATIC`)
  自动完成 bisheng 编译 → host stub 生成 → `ascendc_pack_kernel` 打包 → `libgemm.a`。
- ✅ **NPU 正确性 PASS**:`./ascend_gemm` 输出 `ascend_c GEMM: PASS (max_abs_error=0, M=N=K=128, dtype=fp16)`。
  算法与 `python/` 基准同源(朴素三重循环 + fp32 累加)。

### 构建原理:为什么需要 ascendc.cmake 框架

bisheng 直接编出的 raw `.o` 缺运行时 magic header,`aclrtBinaryLoadFromFile` 直接加载会报 `107000 ACL_ERROR_RT_PARAM_INVALID`。
官方 `ascendc.cmake` 框架自动完成整套打包流程:

1. bisheng `-E` 预处理 kernel → 提取函数签名 → 生成 `host_stub.cpp`
2. bisheng 编译 device kernel → `device_aiv.o` (Vector 核心)
3. `update_host_stub.py` 补丁 `host_stub.cpp` (填充 SoC 专属符号)
4. g++ 编译 `host_stub.cpp` → `host_stub.o`
5. `ascendc_pack_kernel` 把 `device_aiv.o` 合并进 `host_stub.o` → 可被 ACL 运行时加载的二进制
6. `ar` 打包 → `lib/libgemm.a` (导出 `aclrtlaunch_gemm_kernel()`)

最终 host 只需调用 `aclrtlaunch_gemm_kernel(1, stream, d_A, d_B, d_C, nullptr, d_tiling)` 即可启动 kernel,
类比 CUDA 的 `kernel<<<grid,block>>>(...)` launch stub。

> SoC → BUILD_MODE → aicore-arch 映射 (CANN `host_config.cmake`):
> `ascend910b*` → `c220` → `dav-c220-cube`; `ascend310b*` → `m300`; `ascend310p*` → `m200`。
> 旧的 `--cce-soc-version` / `--cce-soc-core-type` 在 CANN 9.0.0 已废弃。

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

# 2. 构建 (ascendc.cmake 自动: 编 kernel + 打包 + 编 host)
cd ascend_c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. 运行 host → NPU 执行 kernel → 正确性校验
cd build && ./ascend_gemm
```

预期输出:
```
ascend_c GEMM: PASS (max_abs_error=0, M=N=K=128, dtype=fp16)
```

若芯片型号不同,可用 cache 变量覆盖:
```bash
cmake -S . -B build -DSOC_VERSION=Ascend910B2   # 默认值
# 其它芯片示例:
#   -DSOC_VERSION=Ascend310B1
#   -DSOC_VERSION=Ascend310P3
```

> **注意**:`CMAKE_BUILD_TYPE` 必须设置 (Release/Debug),`ascendc.cmake` 的 merge 步骤需要它。
> CMakeLists.txt 已设默认值 Release,手动传 `-DCMAKE_BUILD_TYPE=...` 可覆盖。

## 文件说明

| 文件 | 作用 |
|---|---|
| `op_kernel/gemm_kernel.cpp` | Ascend C kernel:朴素三重循环,fp16+fp32 累加,详细注释 |
| `src/gemm_host.cpp` | host 驱动:ACL 初始化、H2D/D2H、调 `aclrtlaunch_gemm_kernel`、CPU 参考+校验 |
| `op_kernel/softmax_kernel.cpp` + `src/softmax_host.cpp` | Softmax:逐行 3-pass(max/exp+sum/normalize),常数从 tiling DataCopy 取 |
| `op_kernel/gelu_kernel.cpp`(+ v3/v5/v6/scalar 变体) | GELU:Vector tile 流水线版 + 各代踩坑诊断版 |
| `op_kernel/rmsnorm_kernel.cpp` + `src/rmsnorm_host.cpp` | RMSNorm:2-pass(fp32 Σx² → TPipe UB 上的 Sqrt → 乘 inv_rms·gamma);常数经 tiling `GlobalTensor.GetValue` 标量读 |
| `op_kernel/rope_kernel.cpp` + `src/rope_host.cpp` | RoPE:交错配对旋转,host 预计算 cos/sin 表 (fp16) 下发,kernel 查表逐对 fp32 乘加 |
| `op_kernel/quant_kernel.cpp` + `src/quant_host.cpp` | INT8 量化/反量化:per-row absmax;int8↔浮点走 Cast 内建 (dav-c220 无 fp32↔int8 直转,经 fp16 中转) |
| `op_kernel/gqa_kernel.cpp` + `src/gqa_host.cpp` | GQA 解码注意力:标量 3-pass (打分→softmax→加权),scratch 存分数 |
| `op_kernel/flash_kernel.cpp` + `src/flash_host.cpp` | FlashAttention:逐行 online softmax 两趟扫描,L×S 分数不整体物化 |
| `CMakeLists.txt` | 构建:引 `ascendc.cmake` → `ascendc_library STATIC` 自动编+打包 kernel,g++ 编 host |

## 运行 (除 GEMM 外的算子)

```bash
cd build
./ascend_softmax 16 512     # 行 softmax
./ascend_rmsnorm 16 512     # RMSNorm (也试 128 4096)
./ascend_rope 16 128        # RoPE (也试 256 128 / 1024 512)
```

> **RMSNorm 踩坑记录 (2026-09)**:① tiling 里的浮点常数若用
> `DataCopy(Cl, Cg, 8)` + 裸 `LocalTensor.GetValue` 读取, 部分 kernel 会被
> 优化丢弃, 改用 `GlobalTensor<float>.GetValue(i)` 标量读最稳;② Sqrt 的
> 工作张量要用 `TPipe/TBuf` 分配真实 UB (裸 LocalTensor 无后备存储);
> ③ host 端 tiling 赋值后别再跑清零循环覆盖 —— `cf[3]=1/D` 被抹零后
> DF=0 → 输出全 0, 现象上极像 kernel/缓存问题, 排查半天实为两行 host 代码。

## 常见问题

- **`ASCEND_HOME_PATH 未设置`**:没 source `set_env.sh`,或在本 shell 重新 source。
- **`SOC_VERSION does not support`**:`-DSOC_VERSION` 值不在支持列表。910B 家族用 `Ascend910B2`(完整列表见 CANN `host_config.cmake` 的 `ascend910b_list`)。
- **`merge_device_obj.py: error: argument --build-type`**:`CMAKE_BUILD_TYPE` 未设。CMakeLists 已设默认 Release;若手动 cmake 不传则需加 `-DCMAKE_BUILD_TYPE=Release`。
- **`fatal error: 'acl/acl.h' file not found`**:CANN include 路径未配置,确认 `set_env.sh` 已 source。
- **`reinterpret_cast from '__gm__ uint8_t *' to 'uint32_t *' is not allowed`**:GM 指针不能直接 cast 到私有指针,需保留 `__gm__` 修饰符(`__gm__ uint32_t*`)。
- **精度误差大**:确认累加器是 `float` 而非 `half`;朴素版逐元素读 GM 慢属正常现象。
