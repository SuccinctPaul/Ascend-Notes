cmake_minimum_required(VERSION 3.16.0)
project(ascend_ops LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CANN 环境变量 ----
# 使用前必须先 source CANN 的 set_env.sh, 它会设置 ASCEND_HOME_PATH
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME)
    message(FATAL_ERROR
        "ASCEND_HOME_PATH 未设置, 请先 source CANN 的 set_env.sh:\n"
        "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- SoC 型号 ----
set(SOC_VERSION "Ascend910B2" CACHE STRING
    "Ascend SoC version (Ascend910B2, Ascend310B1, Ascend310P3, ...)")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME})

# ---- 引入官方 Ascend-C 构建框架 ----
include(${ASCEND_HOME}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# ============================================================================
# 1. GEMM (保留原有功能)
# ============================================================================
ascendc_library(gemm STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gemm_kernel.cpp)

add_executable(ascend_gemm src/gemm_host.cpp)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm PRIVATE gemm ascendcl runtime)

# ============================================================================
# 2. GELU —— Production (Vector tile + DataCopy 流水线版本)
#    kernel: gelu_kernel()  from op_kernel/gelu_kernel.cpp
#    host:   gelu_host.cpp  -> binary ascend_gelu
# ============================================================================
ascendc_library(gelu STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_kernel.cpp)

add_executable(ascend_gelu src/gelu_host.cpp)
target_compile_options(ascend_gelu PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu PRIVATE gelu ascendcl runtime)

# ============================================================================
# 3. GELU —— Scalar 地板性能版本 (故意不用 DataCopy / Vector / Tile)
#    kernel: gelu_scalar_kernel()  from op_kernel/gelu_scalar_kernel.cpp
#    host:   gelu_host.cpp 但带宏 SCALAR_KERNEL=1 切换 launch 函数名
#            -> binary ascend_gelu_scalar
#    注意: 文件目前后缀 .cpp-E (意为"编辑中"), 先 copy 一份无后缀版本再编译.
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_scalar_kernel.cpp)
    set(_GELU_SCALAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_scalar_kernel.cpp)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_scalar_kernel.cpp-E)
    set(_GELU_SCALAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_scalar_kernel.cpp-E)
else()
    message(WARNING "gelu_scalar_kernel.cpp(.cpp-E) not found; skip building ascend_gelu_scalar")
    set(_GELU_SCALAR_SRC "")
endif()

if(_GELU_SCALAR_SRC)
    ascendc_library(gelu_scalar STATIC ${_GELU_SCALAR_SRC})

    add_executable(ascend_gelu_scalar src/gelu_host.cpp)
    target_compile_definitions(ascend_gelu_scalar PRIVATE SCALAR_KERNEL=1)
    target_compile_options(ascend_gelu_scalar PRIVATE -fno-fast-math)
    target_link_libraries(ascend_gelu_scalar PRIVATE gelu_scalar gelu ascendcl runtime)

    # ascend_gelu 需要同时链接 scalar stub (host 运行期 argv[2]=="scalar" 分支会调用
    # aclrtlaunch_gelu_scalar_kernel, 即便默认不跑该分支, 静态链接也需要符号解析)
    target_link_libraries(ascend_gelu PRIVATE gelu_scalar)
endif()

# ============================================================================
# 4. Softmax —— Production (UB tile + Reduce/Exp/Div Vector 版本)
#    kernel: softmax_kernel()  from op_kernel/softmax_kernel.cpp
#    host:   softmax_host.cpp   -> binary ascend_softmax
# ============================================================================
ascendc_library(softmax STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/softmax_kernel.cpp)

add_executable(ascend_softmax src/softmax_host.cpp)
target_compile_options(ascend_softmax PRIVATE -fno-fast-math)
target_link_libraries(ascend_softmax PRIVATE softmax ascendcl runtime)

# ============================================================================
# 5. Softmax —— Scalar 地板性能版本
#    kernel: softmax_scalar_kernel()  from op_kernel/softmax_scalar_kernel.cpp
#    host:   softmax_host.cpp 带宏 SCALAR_KERNEL=1 -> ascend_softmax_scalar
# ============================================================================
ascendc_library(softmax_scalar STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/softmax_scalar_kernel.cpp)

add_executable(ascend_softmax_scalar src/softmax_host.cpp)
target_compile_definitions(ascend_softmax_scalar PRIVATE SCALAR_KERNEL=1)
target_compile_options(ascend_softmax_scalar PRIVATE -fno-fast-math)
# softmax_host 两个 extern 函数声明都会存在 (运行期 argv[3]=="scalar" 切换),
# 所以 ascend_softmax / ascend_softmax_scalar 两个 binary 都同时链接两个 kernel stub.
target_link_libraries(ascend_softmax_scalar PRIVATE softmax_scalar softmax ascendcl runtime)
target_link_libraries(ascend_softmax        PRIVATE softmax_scalar)

# ============================================================================
# 8. RMSNorm —— 标量规范版 (Σx² fp32 归约 + sqrt + 乘倒数缩放)
#    kernel: rmsnorm_kernel()  from op_kernel/rmsnorm_kernel.cpp
#    host:   rmsnorm_host.cpp  -> binary ascend_rmsnorm
# ============================================================================
ascendc_library(rmsnorm STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/rmsnorm_kernel.cpp)

add_executable(ascend_rmsnorm src/rmsnorm_host.cpp)
target_compile_options(ascend_rmsnorm PRIVATE -fno-fast-math)
target_link_libraries(ascend_rmsnorm PRIVATE rmsnorm ascendcl runtime)

# ============================================================================
# 9. RoPE —— 交错配对旋转 (查预计算 cos/sin 表, 纯逐元素乘加)
#    kernel: rope_kernel()  from op_kernel/rope_kernel.cpp
#    host:   rope_host.cpp  -> binary ascend_rope
# ============================================================================
ascendc_library(rope STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/rope_kernel.cpp)

add_executable(ascend_rope src/rope_host.cpp)
target_compile_options(ascend_rope PRIVATE -fno-fast-math)
target_link_libraries(ascend_rope PRIVATE rope ascendcl runtime)

# ============================================================================
# 7. GELU v3 —— 全新命名 target (gelu_v3_kernel / gelu_v3_host) 避免 CANN 9.0
#    ascendc_library ExternalProject 对旧名字 gelu / gelu_scalar 的二进制打包缓存导致
#    kernel 源码变更即便 --clean-first 也不会重新 pack 进可执行文件。
# ============================================================================
ascendc_library(gelu_v3 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_v3_kernel.cpp)
add_executable(ascend_gelu_v3 src/gelu_v3_host.cpp)
target_compile_options(ascend_gelu_v3 PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu_v3 PRIVATE gelu_v3 ascendcl runtime)

# v5: 常数从 tiling 通过 DataCopy 搬到 UB LocalTensor (绕开 SetValue(立即数) → -inf bug)
ascendc_library(gelu_v5 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_v5_kernel.cpp)
add_executable(ascend_gelu_v5 src/gelu_v5_host.cpp)
target_compile_options(ascend_gelu_v5 PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu_v5 PRIVATE gelu_v5 ascendcl runtime)

# v6: DataCopy 常数 + 完全复刻 softmax 内循环 AST (sBIG 不再 roundtrip; big 直接 local var)
ascendc_library(gelu_v6 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_v6_kernel.cpp)
add_executable(ascend_gelu_v6 src/gelu_v5_host.cpp)
target_compile_definitions(ascend_gelu_v6 PRIVATE USE_KERNEL_V6=1)
target_compile_options(ascend_gelu_v6 PRIVATE -fno-fast-math)
target_link_libraries(ascend_gelu_v6 PRIVATE gelu_v6 ascendcl runtime)

# ============================================================================
# 6. 诊断用小工具 (旧版本保留, 不影响核心构建)
# ============================================================================
foreach(_d diag diag2 diag3 diag4 diag5 diag6 diag_gelu gelu_min identity identity2 identity3 identity_s0 diag_block identity_fix identity_chunk identity_mod)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/${_d}_kernel.cpp AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${_d}_host.cpp)
        ascendc_library(${_d} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/${_d}_kernel.cpp)
        add_executable(ascend_${_d} src/${_d}_host.cpp)
        target_compile_options(ascend_${_d} PRIVATE -fno-fast-math)
        target_link_libraries(ascend_${_d} PRIVATE ${_d} ascendcl runtime)
    endif()
endforeach()
