# This file is distributed under the MIT license.
# See the LICENSE file for details.

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)
set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/imgui)
set(IMGUI_INCLUDE_DIR ${IMGUI_DIR})


#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

# Ptex

if (VSNRAY_ENABLE_PTEX)
    find_package(Ptex)
    find_package(ZLIB)
    visionaray_use_package(Ptex)
    visionaray_use_package(ZLIB)
endif()

# Qt5

if (VSNRAY_ENABLE_QT5)
    find_package(Qt5Core)
    find_package(Qt5Gui)
    find_package(Qt5OpenGL)
    find_package(Qt5Widgets)
    visionaray_use_package(Qt5Core)
    visionaray_use_package(Qt5Gui)
    visionaray_use_package(Qt5OpenGL)
    visionaray_use_package(Qt5Widgets)
endif()

# SDL2

if (VSNRAY_ENABLE_SDL2)
    find_package(SDL2)
    visionaray_use_package(SDL2)
endif()

# TBB

if (VSNRAY_ENABLE_TBB)
    find_package(TBB)
    visionaray_use_package(TBB)
endif()


#--------------------------------------------------------------------------------------------------
#
#

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})
include_directories(${IMGUI_INCLUDE_DIR})

set(VIEWER_HEADERS
    call_kernel.h
    host_device_rt.h
)

set(VIEWER_SOURCES
    render_generic_material.cpp
    render_instances.cpp
    render_instances_ptex.cpp
    render_plastic.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(VIEWER_CUDA_SOURCES
        host_device_rt.cu
        render_generic_material.cu
        render_instances.cu
        render_plastic.cu
        viewer.cu
    )
else()
    set(VIEWER_SOURCES ${VIEWER_SOURCES}
        host_device_rt.cpp
        viewer.cpp
    )
endif()


#--------------------------------------------------------------------------------------------------
# Add viewer target
#

visionaray_add_executable(viewer
    ${VIEWER_HEADERS}
    ${VIEWER_SOURCES}
    ${VIEWER_CUDA_SOURCES}
)


#--------------------------------------------------------------------------------------------------
# Install viewer
#

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/viewer
    DESTINATION bin
    RENAME vsnray-viewer
)
