cmake_minimum_required(VERSION 3.16.0)
project(ascend_ops LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CANN 环境变量 ----
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME)
    message(FATAL_ERROR
        "ASCEND_HOME_PATH 未设置, 请先 source CANN 的 set_env.sh:\n"
        "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- SoC 型号 (可被 -D 覆盖) ----
set(SOC_VERSION "Ascend910B2" CACHE STRING
    "Ascend SoC version (Ascend910B2, Ascend310B1, Ascend310P3, ...)")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME})

# ---- 引入官方 Ascend-C 构建框架 ----
include(${ASCEND_HOME}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# =====================================================================
# GEMM  (原 gemm_kernel.cpp, 已验证)
# =====================================================================
ascendc_library(gemm STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gemm_kernel.cpp)

add_executable(ascend_gemm src/gemm_host.cpp)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm PRIVATE gemm ascendcl runtime)

# =====================================================================
# GELU (生产版) —— AscendC adv_api Gelu<half, highPrecision=true> + DataCopy
# =====================================================================
# kernel 入口: gelu_kernel(x, y, workspace, tiling)
# workspace 不用传 nullptr; tiling 放 N (元素数, uint32_t).
ascendc_library(gelu STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_kernel.cpp)

add_executable(ascend_gelu src/gelu_host.cpp)
target_compile_options(ascend_gelu PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu PRIVATE gelu ascendcl runtime)

# =====================================================================
# GELU (教学版 scalar) —— 逐元素 scalar GM, ≤2 层 fp32 显式拆分
# 用于展示 "scalar 地板性能" (~1 GB/s, HBM util < 0.1%)
# =====================================================================
ascendc_library(gelu_scalar STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_scalar_kernel.cpp)

add_executable(ascend_gelu_scalar src/gelu_scalar_host.cpp)
target_compile_options(ascend_gelu_scalar PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu_scalar PRIVATE gelu_scalar ascendcl runtime)

# =====================================================================
# Diag2: LocalTensor + DataCopy + Muls(y = 0.5*x), 验证 Vector 基础设施
# =====================================================================
ascendc_library(diag2 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/diag2_kernel.cpp)

add_executable(ascend_diag2 src/diag2_host.cpp)
target_compile_options(ascend_diag2 PRIVATE -fno-fast-math)
target_link_libraries(ascend_diag2 PRIVATE diag2 ascendcl runtime)

# =====================================================================
# Diag4: 纯标量 GM↔UB, 全 fp32 Vector GELU, 消除 DataCopy 计数歧义
# =====================================================================
ascendc_library(diag4 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/diag4_kernel.cpp)

add_executable(ascend_diag4 src/diag4_host.cpp)
target_compile_options(ascend_diag4 PRIVATE -fno-fast-math)
target_link_libraries(ascend_diag4 PRIVATE diag4 ascendcl runtime)

