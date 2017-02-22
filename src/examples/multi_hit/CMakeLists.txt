# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLUT REQUIRED)

if(VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLUT)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(EX_MULTI_HIT_CUDA_SOURCES
        main.cu
    )
else()
    set(EX_MULTI_HIT_SOURCES
        main.cpp
    )
endif()

visionaray_add_executable(multi_hit
    ${EX_MULTI_HIT_SOURCES}
    ${EX_MULTI_HIT_CUDA_SOURCES}
)
