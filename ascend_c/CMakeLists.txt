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

# ---- 查找 ascendc 编译器 (kernel DSL 编译驱动) ----
# set_env.sh 会把 ascendc 加入 PATH; 兼容性回退到 CANN 的 compiler/bin 目录
find_program(ASCENDC_COMPILER
    NAMES ascendc
    PATHS
        ${ASCEND_HOME}/bin
        ${ASCEND_HOME}/compiler
        ${ASCEND_HOME}/aarch64-linux/bin
    REQUIRED)

# ---- 芯片型号 ----
# 本机为 Ascend910B 系列; 若编译报芯片不匹配, 改为 Ascend910B3 / Ascend910B4
set(ASCEND_CHIP "Ascend910B2" CACHE STRING "Ascend chip variant")

# ---- Ascend C kernel 编译 ----
# ascendc 把 C++ kernel 源码编译成 device 可执行的 .o
add_custom_command(
    OUTPUT gemm_kernel.o
    COMMAND ${ASCENDC_COMPILER} --chip=${ASCEND_CHIP}
            ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
            -o gemm_kernel.o
    DEPENDS ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
    COMMENT "Compiling Ascend C kernel (chip=${ASCEND_CHIP})"
    VERBATIM
)
add_custom_target(kernel ALL DEPENDS gemm_kernel.o)

# ---- host 侧可执行文件 ----
# aarch64 GCC 原生支持 __fp16; -fno-fast-math 避免过度优化影响精度校验
add_executable(ascend_gemm src/gemm_host.cpp)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm ascendcl acl)
