cmake_minimum_required(VERSION 3.10)
project(ascend_gemm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CANN 环境变量 ----
# 使用前必须先 source CANN 的 set_env.sh, 它会设置 ASCEND_HOME_PATH
# (例如 /usr/local/Ascend/cann-9.0.0)
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME)
    message(FATAL_ERROR
        "ASCEND_HOME_PATH 未设置, 请先 source CANN 的 set_env.sh:\n"
        "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh")
endif()

include_directories(${ASCEND_HOME}/include)
link_directories(${ASCEND_HOME}/lib64)

# ---- 查找 Ascend C kernel 编译器 ----
# 实测(CANN 9.0.0, aarch64): 此版本没有名为 `ascendc` 的可执行文件,
# `$ASCEND_HOME/compiler/ascendc` 是一个含 include/ 的目录(符号链接);
# 真正的 kernel 编译器是 `bisheng` (CANN 的 ASC 编译器前端, 即 bisheng/毕昇),
# 由 `$ASCEND_HOME/bin/bisheng` 提供 (source set_env.sh 后在 PATH 中).
# 官方 Ascend-C 构建框架 (ascendc_kernel_cmake) 也是 find_program(... NAMES bisheng ...).
find_program(ASCENDC_COMPILER
    NAMES bisheng
    PATHS
        ${ASCEND_HOME}/bin
        ${ASCEND_HOME}/aarch64-linux/ccec_compiler/bin
        ${ASCEND_HOME}/compiler/ccec_compiler/bin
    REQUIRED)

# Ascend-C kernel 头文件 (kernel_operator.h 等): 在 aarch64-linux/ascendc/include
set(ASCENDC_INCLUDE ${ASCEND_HOME}/aarch64-linux/ascendc/include)

# ---- 芯片型号 / soc 标志 (可被 -D 覆盖, 便于在远程调参) ----
# 远程 NPU: npu-smi 显示 910B2; CANN 的 SOC_MAP_EXT 把 ascend910b 家族映射到 soc_version=Ascend910B1.
# bisheng 接受:  --cce-soc-version=<soc>  与  --cce-soc-core-type=<core>
# 实测 910B1/B2 上, 直接给 --cce-soc-core-type=AiCore/AICore/MIX/... 均被拒
# (报 "soc_core_type: X is not supported for soc_version Ascend910B1"),
# 正确的 core_type 组合需在远程用官方 ascendc.cmake 框架确认 (见 README "构建状态").
# 因此这里把 soc_version / soc_core_type 暴露为 cache 变量, 默认只传 soc_version.
set(ASCEND_SOC_VERSION "Ascend910B1" CACHE STRING "bisheng --cce-soc-version value (910B family = Ascend910B1)")
set(ASCEND_SOC_CORE_TYPE "" CACHE STRING "bisheng --cce-soc-core-type value (empty=不传, 待远程确认)")

# ---- Ascend C kernel 编译 ----
# --asc-aicore-lang: 把输入当作 ASC (Ascend C) 语言处理 (官方框架的写法)
# --cce-soc-version: 指定目标芯片 (soc_version)
# (可选) --cce-soc-core-type: 指定核类型, 当前留空 —— 见上方注释
set(_SOC_FLAGS --asc-aicore-lang --cce-soc-version=${ASCEND_SOC_VERSION})
if(ASCEND_SOC_CORE_TYPE)
    list(APPEND _SOC_FLAGS --cce-soc-core-type=${ASCEND_SOC_CORE_TYPE})
endif()

add_custom_command(
    OUTPUT gemm_kernel.o
    COMMAND ${ASCENDC_COMPILER} ${_SOC_FLAGS}
            -I${ASCENDC_INCLUDE}
            ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
            -c -o gemm_kernel.o
    DEPENDS ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
    COMMENT "Compiling Ascend C kernel (bisheng, soc=${ASCEND_SOC_VERSION}${ASCEND_SOC_CORE_TYPE})"
    VERBATIM
)
add_custom_target(kernel ALL DEPENDS gemm_kernel.o)

# ---- host 侧可执行文件 ----
# aarch64 GCC 原生支持 __fp16; -fno-fast-math 避免过度优化影响精度校验
add_executable(ascend_gemm src/gemm_host.cpp)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm ascendcl acl)
