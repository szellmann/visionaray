project(visionaray)


cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
    "${PROJECT_SOURCE_DIR}/cmake/modules"
)

include(VisionarayAddTarget)
include(VisionarayUseFramework)
include(VisionarayUsePackage)


#---------------------------------------------------------------------------------------------------
# Visionaray Options
#

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" OFF)
option(VSNRAY_ENABLE_WARNINGS "Enable all warnings" ON)
option(VSNRAY_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
option(VSNRAY_ENABLE_COVER "Build OpenCOVER plugin" ON)
option(VSNRAY_ENABLE_CUDA "Use CUDA, if available" ON)
option(VSNRAY_ENABLE_EXAMPLES "Build the programming examples" OFF)
option(VSNRAY_ENABLE_QT5 "Use Qt5, if available" ON)
option(VSNRAY_ENABLE_VIEWER "Build the vsnray-viewer program" ON)

#---------------------------------------------------------------------------------------------------
# All VSNRAY_* options have to be specified before this include
#

include(VisionarayHandleOptions)

add_subdirectory(src)
