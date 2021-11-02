# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

if (VSNRAY_ENABLE_TBB)
    find_package(TBB)
    visionaray_use_package(TBB)
endif()

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/imgui)
set(IMGUI_INCLUDE_DIR ${IMGUI_DIR})

set(RAPID_JSON_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/rapidjson)
set(RAPID_JSON_INCLUDE_DIR ${RAPID_JSON_DIR}/include)

set(TINYPLY_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/tinyply)
set(TINYPLY_INCLUDE_DIR ${TINYPLY_DIR}/source)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})

add_subdirectory(ao)
add_subdirectory(cuda_unified_memory)
add_subdirectory(generic_primitive)
add_subdirectory(intersector)
add_subdirectory(juggler)
#add_subdirectory(minimal)
#add_subdirectory(multi_hit)
add_subdirectory(multi_volume)
if (VSNRAY_ENABLE_CUDA AND CUDA_FOUND)
add_subdirectory(opengl_interop)
endif()
add_subdirectory(phantom)
add_subdirectory(raytracinginoneweekend)
if (VSNRAY_ENABLE_CUDA AND CUDA_FOUND)
add_subdirectory(raytracinginoneweekend_cuda)
endif()
add_subdirectory(smallpt)
add_subdirectory(texture3d)
add_subdirectory(volume)
