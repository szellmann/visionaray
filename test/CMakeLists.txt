# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(VSNRAY_ENABLE_UNITTESTS)
    add_subdirectory(unittests)
endif()

# HIP test
if(VSNRAY_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(rocthrust REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(Threads REQUIRED)

  add_executable(hip_test hip_test.hip)
  set_target_properties(hip_test PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    LINKER_LANGUAGE HIP
  )
  target_include_directories(hip_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../src/visionaray
  )
  target_link_libraries(hip_test PRIVATE
    hip::host
    hip::device
    roc::rocthrust
    hip::hipcub
    Threads::Threads
  )
  add_test(NAME hip_test COMMAND hip_test)

  add_executable(hip_random_test hip_random_test.hip)
  set_target_properties(hip_random_test PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    LINKER_LANGUAGE HIP
  )
  target_include_directories(hip_random_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../src/visionaray
  )
  target_link_libraries(hip_random_test PRIVATE
    hip::host
    hip::device
    roc::rocthrust
    hip::hipcub
    Threads::Threads
  )
  add_test(NAME hip_random_test COMMAND hip_random_test)
endif()
