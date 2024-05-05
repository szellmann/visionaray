# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(NOT VSNRAY_ENABLE_CUDA)
    message(STATUS "Not building cuda_unified_memory example, CUDA builds not enabled") 
    return()
endif()

if(APPLE)
    message(STATUS "Not building cuda_unified_memory example, not supported on macOS")
    return()
endif()

enable_language(CUDA)
add_executable(cuda_unified_memory)
target_sources(cuda_unified_memory PRIVATE main.cu)
target_link_libraries(cuda_unified_memory PUBLIC visionaray visionaray_common)
