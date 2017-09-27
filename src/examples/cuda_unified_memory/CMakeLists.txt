# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(NOT VSNRAY_ENABLE_CUDA)
    message(STATUS "Not building cuda_unified_memory example, CUDA builds not enabled") 
    return()
endif()

if(APPLE)
    message(STATUS "Not building cuda_unified_memory example, not supported on macOS")
endif()

find_package(CUDA)

if(NOT CUDA_FOUND)
    message(STATUS "Not building cuda_unified_memory example, CUDA not found") 
    return()
endif()

visionaray_cuda_compile(EX_CUDA_UNIFIED_MEMORY_SOURCES main.cu)

visionaray_add_executable(cuda_unified_memory
    ${EX_CUDA_UNIFIED_MEMORY_SOURCES}
)
