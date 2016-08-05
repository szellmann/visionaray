project(visionaray)


cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
    "${PROJECT_SOURCE_DIR}/cmake/modules"
)

include(CTest)
enable_testing()
include(VisionarayCompileFailureTest)

include(VisionarayAddTarget)
include(VisionarayUseFramework)
include(VisionarayUsePackage)


#--------------------------------------------------------------------------------------------------
# CMake policies to control backward compatible behavior
#

# Old behavior: COMPILE_DEFINITIONS_<CONFIG>
# New behavior: COMPILE_DEFINITIONS + generator expression
if(POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif()


#---------------------------------------------------------------------------------------------------
# Visionaray Options
#

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" OFF)
option(VSNRAY_ENABLE_WARNINGS "Enable all warnings" ON)
option(VSNRAY_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
option(VSNRAY_ENABLE_CUDA "Use CUDA, if available" ON)
option(VSNRAY_ENABLE_EXAMPLES "Build the programming examples" OFF)
option(VSNRAY_ENABLE_QT5 "Use Qt5, if available" OFF)
option(VSNRAY_ENABLE_VIEWER "Build the vsnray-viewer program" ON)
option(VSNRAY_ENABLE_UNITTESTS "Build unit tests" OFF)


#---------------------------------------------------------------------------------------------------
# All VSNRAY_* options have to be specified before this include
#

include(VisionarayHandleOptions)

add_subdirectory(src)
add_subdirectory(test)
