# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_add_cuda_executable(multi_hit main.cu EXCLUDE_FROM_ALL)
else()
    visionaray_add_executable(multi_hit main.cpp EXCLUDE_FROM_ALL)
endif()
