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
    visionaray_cuda_compile(EX_SMALLPT_CUDA_SOURCES
        main.cu
    )
else()
    set(EX_SMALLPT_SOURCES
        main.cpp
    )
endif()

visionaray_add_executable(smallpt
    ${EX_SMALLPT_SOURCES}
    ${EX_SMALLPT_CUDA_SOURCES}
)
