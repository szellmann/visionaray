# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(GLUT REQUIRED)
find_package(TBB REQUIRED)

visionaray_use_package(GLUT)
visionaray_use_package(TBB)

visionaray_use_package(GLUT)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(EX_WAVEFRONT_PATHTRACER_CUDA_SOURCES
        main.cu
    )
else()
    set(EX_WAVEFRONT_PATHTRACER_SOURCES
        main.cpp
    )
endif()

visionaray_add_executable(wavefront_pathtracer
    ${EX_WAVEFRONT_PATHTRACER_SOURCES}
    ${EX_WAVEFRONT_PATHTRACER_CUDA_SOURCES}
)
