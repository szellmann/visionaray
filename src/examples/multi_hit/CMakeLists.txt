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
    visionaray_add_cuda_executable(multi_hit main.cu)
else()
    visionaray_add_executable(multi_hit main.cpp)
endif()

# Don't compile due to issues with simd overload of insert_sorted()
set_target_properties(multi_hit PROPERTIES EXCLUDE_FROM_ALL TRUE)
