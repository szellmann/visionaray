# This file is distributed under the MIT license.
# See the LICENSE file for details.

cmake_minimum_required(VERSION 3.27)

if (TARGET visionaray::visionaray)
  return()
endif()

macro(visionaray_config_message)
  if (NOT DEFINED visionaray_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()

## Setup base target ##

add_library(visionaray::visionaray INTERFACE IMPORTED)
target_include_directories(visionaray::visionaray INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../include)

find_dependency(Threads)
find_package(CUDAToolkit QUIET)
find_package(TBB QUIET)

visionaray_config_message(STATUS "Found visionaray: ${CMAKE_CURRENT_LIST_DIR}")
visionaray_config_message("    --> visionaray::visionaray target available (visionaray::visionaray)")

