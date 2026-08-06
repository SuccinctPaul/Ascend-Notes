cmake_minimum_required(VERSION 3.10)
project(ascend_gemm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
include_directories(${ASCEND_HOME}/include)
link_directories(${ASCEND_HOME}/lib64)

# Ascend-C kernel 编译
add_custom_command(
    OUTPUT gemm_kernel.o
    COMMAND ascendc --chip=Ascend910B2
            ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
            -o gemm_kernel.o
    DEPENDS ${CMAKE_SOURCE_DIR}/op_kernel/gemm_kernel.cpp
)

add_custom_target(kernel ALL DEPENDS gemm_kernel.o)

# Host 侧
add_executable(ascend_gemm src/gemm_host.cpp)
target_link_libraries(ascend_gemm acl ascendcl)
