install(
    TARGETS ascend_c_exe
    RUNTIME COMPONENT ascend_c_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
