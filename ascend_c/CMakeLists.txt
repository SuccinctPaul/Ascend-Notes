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

# Ascend-C kernel 头文件 (kernel_operator.h 等)。
# 官方 bisheng_intf.cmake 用一组 asc/include + asc/impl + tikcpp/tikcfw 路径,
# 这里取其并集以保证 kernel_operator.h / GlobalTensor 等均可解析。
set(ASCENDC_DEVKIT ${ASCEND_HOME}/aarch64-linux)
set(ASCENDC_INCLUDE
    ${ASCENDC_DEVKIT}/asc/include
    ${ASCENDC_DEVKIT}/asc/include/basic_api
    ${ASCENDC_DEVKIT}/asc/impl
    ${ASCENDC_DEVKIT}/asc/impl/basic_api
    ${ASCENDC_DEVKIT}/tikcpp/tikcfw)

# ---- 芯片型号 / aicore-arch 标志 (可被 -D 覆盖, 便于在远程调参) ----
# 远程 NPU: npu-smi 显示 910B2。
# CANN 9.0.0 的 ascendc.cmake 框架 (legacy_modules/host_config.cmake) 把
#   ascend910b* 家族 → BUILD_MODE=c220
# 再由 bisheng_intf.cmake 把 c220 映射到 arch:
#   - dav-c220-cube: AI Core (含 Cube + Vector, 矩阵乘走这里)
#   - dav-c220-vec : 纯 Vector Core (AIV)
# 本 GEMM kernel 跑在 AI Core 上, 故用 dav-c220-cube。
# 旧的 --cce-soc-version / --cce-soc-core-type 标志在 CANN 9.0.0 上已不接受
# (报 "soc_core_type: X is not supported for soc_version Ascend910B1")。
set(ASCEND_AICORE_ARCH "dav-c220-cube" CACHE STRING
    "bisheng --cce-aicore-arch value (ascend910b2 → dav-c220-cube; ascend310b → dav-m300; ...)")

# ---- Ascend C kernel 编译 ----
# --cce-aicore-lang : 把输入当作 ASC (Ascend C) 语言处理 (旧名 --asc-aicore-lang)
# --cce-aicore-arch : 指定目标 AI Core 微架构 (替代旧的 --cce-soc-version)
# --cce-aicore-only : 只编 device kernel, 不链接 host stub
# --cce-auto-sync   : 自动插入必要的同步 (官方框架默认开)
# --cce-mask-opt    : mask 优化 (cube 模式官方默认开, 这里也带上)
#
# 注意: bisheng 在 --cce-aicore-only 模式下不接受 -o (报 "cannot specify -o
# when generating multiple output files")。它会把 .o 写到 cwd, 文件名带 arch 后缀,
# 形如 gemm_kernel-cce-hiipu64-hisilicon-cce-dav-c220-cube.o。因此先把输出目录
# 指向 ${KERNEL_OBJ_DIR}, 编完再用 shell 把唯一那个 .o 移成 gemm_kernel.o。
set(KERNEL_OBJ_DIR ${CMAKE_BINARY_DIR}/kernel_obj)
file(MAKE_DIRECTORY ${KERNEL_OBJ_DIR})
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gemm_kernel.o
    COMMAND ${ASCENDC_COMPILER}
            --cce-aicore-lang
            --cce-aicore-arch=${ASCEND_AICORE_ARCH}
            --cce-aicore-only
            --cce-auto-sync
            --cce-mask-opt
            -std=c++17 -O3
            ${ASCENDC_INCLUDE}
            ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
            -c
    COMMAND bash -c "mv ${KERNEL_OBJ_DIR}/gemm_kernel-*.o ${CMAKE_BINARY_DIR}/gemm_kernel.o"
    WORKING_DIRECTORY ${KERNEL_OBJ_DIR}
    DEPENDS ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
    COMMENT "Compiling Ascend C kernel (bisheng, arch=${ASCEND_AICORE_ARCH})"
    VERBATIM
)
add_custom_target(kernel ALL DEPENDS ${CMAKE_BINARY_DIR}/gemm_kernel.o)

# ---- host 侧可执行文件 ----
# aarch64 GCC 原生支持 __fp16; -fno-fast-math 避免过度优化影响精度校验
# 依赖 kernel target: 确保 ./ascend_gemm 运行前 gemm_kernel.o 已编出 (host 默认从 cwd 加载它)
# 链接库:
#   ascendcl : ACL 主库 (aclrtBinaryLoadFromFile / aclrtLaunchKernelWithConfig 等)
#   acl      : ACL 基础库 (aclInit / aclrtMalloc 等)
#   runtime  : 底层 rt 库 (rtGetC2cCtrlAddr, 910B cube kernel 的 FFTS 地址查询)
add_executable(ascend_gemm src/gemm_host.cpp)
add_dependencies(ascend_gemm kernel)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm ascendcl acl runtime)

# 默认从 build 目录运行: gemm_kernel.o 与 ascend_gemm 同在 ${CMAKE_BINARY_DIR}
# 运行: cd build && ./ascend_gemm   (或 ./ascend_gemm /path/to/gemm_kernel.o)
