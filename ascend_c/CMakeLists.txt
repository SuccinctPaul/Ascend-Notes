cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    ascend_c
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://example.com/"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    ascend_c_lib OBJECT
    source/lib.cpp
)

target_include_directories(
    ascend_c_lib ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(ascend_c_lib PUBLIC cxx_std_17)

# ---- Declare executable ----

add_executable(ascend_c_exe source/main.cpp)
add_executable(ascend_c::exe ALIAS ascend_c_exe)

set_property(TARGET ascend_c_exe PROPERTY OUTPUT_NAME ascend_c)

target_compile_features(ascend_c_exe PRIVATE cxx_std_17)

target_link_libraries(ascend_c_exe PRIVATE ascend_c_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT ascend_c_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of ascend_c"
  )
endif()

include(cmake/dev-mode.cmake)
