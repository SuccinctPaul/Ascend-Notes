# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(ascend_cTests LANGUAGES CXX)

# ---- Tests ----

add_executable(ascend_c_test source/ascend_c_test.cpp)
target_link_libraries(ascend_c_test PRIVATE ascend_c_lib)
target_compile_features(ascend_c_test PRIVATE cxx_std_17)

add_test(NAME ascend_c_test COMMAND ascend_c_test)

# ---- End-of-file commands ----

add_folders(Test)
