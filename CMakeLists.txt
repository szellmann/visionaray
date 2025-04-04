# This file is distributed under the MIT license.
# See the LICENSE file for details.

include(CMakePackageConfigHelpers)
cmake_minimum_required(VERSION 3.22)
project(visionaray VERSION 0.5.1 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_SOURCE_DIR}/cmake/modules
)

include(CTest)
enable_testing()


#--------------------------------------------------------------------------------------------------
# CMake policies to control backward compatible behavior
#

# Old behavior: COMPILE_DEFINITIONS_<CONFIG>
# New behavior: COMPILE_DEFINITIONS + generator expression
if(POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif()


#---------------------------------------------------------------------------------------------------
# Directories containing config headers
#

# Config file location (#include <visionaray/config.h>,#include <common/config.h>,...)
set(__VSNRAY_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/config")

# Config file location for the public visionaray library
set(__VSNRAY_VISIONARAY_CONFIG_DIR "${__VSNRAY_CONFIG_DIR}/visionaray")

# Config file location for the private visionaray_common library
set(__VSNRAY_COMMON_CONFIG_DIR "${__VSNRAY_CONFIG_DIR}/common")


#---------------------------------------------------------------------------------------------------
# Visionaray Options
#

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" OFF)
option(VSNRAY_ENABLE_WARNINGS "Enable all warnings" ON)
option(VSNRAY_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
option(VSNRAY_ENABLE_3DCONNEXIONCLIENT "Use 3DconnexionClient, if available" OFF)
option(VSNRAY_ENABLE_COCOA "Use Cocoa, if available" OFF)
option(VSNRAY_ENABLE_COMMON "Build the common library with several utils" ON)
option(VSNRAY_ENABLE_CUDA "Use CUDA, if available" ON)
option(VSNRAY_ENABLE_EXAMPLES "Build the programming examples" OFF)
option(VSNRAY_ENABLE_PBRT_PARSER "Build with pbrtParser" OFF)
option(VSNRAY_ENABLE_PTEX "Use Ptex, if available" OFF)
option(VSNRAY_ENABLE_QT5 "Use Qt5, if available" OFF)
option(VSNRAY_ENABLE_SDL2 "Use SDL2, if available" OFF)
option(VSNRAY_ENABLE_TBB "Use TBB, if available" OFF)
option(VSNRAY_ENABLE_VIEWER "Build the vsnray-viewer program" ON)
option(VSNRAY_ENABLE_UNITTESTS "Build unit tests" OFF)
option(VSNRAY_MACOSX_BUNDLE "Build executables as application bundles on macOS" ON)
option(VSNRAY_ENABLE_CUDA_STYLE_THREAD_INTROSPECTION "Define CUDA-style thread introspection variables in CPU scheduler headers" OFF)


#---------------------------------------------------------------------------------------------------
# Only add targets and subdirs after all  VSNRAY_* options have been specified
#

add_subdirectory(src)
add_subdirectory(test)
