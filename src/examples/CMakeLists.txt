# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

visionaray_use_package(Boost)
if (CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)
set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})

add_subdirectory(ao)
add_subdirectory(ao_gpu)
add_subdirectory(generic_primitive)
add_subdirectory(intersector)
add_subdirectory(minimal)
add_subdirectory(multi_hit)
add_subdirectory(multi_volume)
add_subdirectory(texture3d)
add_subdirectory(volume)
