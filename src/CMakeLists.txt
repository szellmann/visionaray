# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(NOT BUILD_SHARED_LIBS)
    add_definitions(-DVSNRAY_STATIC)
endif()

add_subdirectory(common)

if(VSNRAY_ENABLE_EXAMPLES)
add_subdirectory(examples)
endif()

if(VSNRAY_ENABLE_VIEWER)
add_subdirectory(viewer)
endif()

add_subdirectory(visionaray)
