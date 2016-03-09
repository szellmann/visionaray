# These macros define CMake tests (for ctest) which can test whether a certain
# code does *not* compile.
#
# Usage:
# visionaray_test_compile_failure(SOURCE_FILE TESTNAME [COMPILE_DEFINITIONS...])
#
# Where COMPILE_DEFINITIONS can be used to pack multiple tests into one source
# file.
#
# visionaray_test_compile_success can be used to test a reference code.

macro(visionaray_test_compile_failure FILENAME TESTNAME)
    add_executable(cft_${TESTNAME} ${FILENAME})
    set_target_properties(cft_${TESTNAME} PROPERTIES
                          EXCLUDE_FROM_ALL TRUE
                          EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    target_compile_definitions(cft_${TESTNAME} PRIVATE ${ARGN})
    add_test(NAME ${TESTNAME}
             COMMAND ${CMAKE_COMMAND} --build . --target cft_${TESTNAME} --config $<CONFIGURATION>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${TESTNAME} PROPERTIES WILL_FAIL TRUE)
endmacro()

macro(visionaray_test_compile_success FILENAME TESTNAME)
    add_executable(cft_${TESTNAME} ${FILENAME})
    set_target_properties(cft_${TESTNAME} PROPERTIES
                          EXCLUDE_FROM_ALL TRUE
                          EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    target_compile_definitions(cft_${TESTNAME} PRIVATE ${ARGN})
    add_test(NAME ${TESTNAME}
             COMMAND ${CMAKE_COMMAND} --build . --target cft_${TESTNAME} --config $<CONFIGURATION>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endmacro()


