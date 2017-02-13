if(VSNRAY_ENABLE_COMPILE_FAILURE_TESTS)
    add_subdirectory(compile_failure_tests)
endif()

if(VSNRAY_ENABLE_UNITTESTS)
    add_subdirectory(unittests)
endif()
