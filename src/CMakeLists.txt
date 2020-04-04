# This file is distributed under the MIT license.
# See the LICENSE file for details.

if(NOT BUILD_SHARED_LIBS)
    add_definitions(-DVSNRAY_STATIC)
endif()

if(APPLE AND VSNRAY_ENABLE_CUDA)
    # Work around an incompatibility between
    # boost::std_category and std::error_category
    add_definitions(-DBOOST_NO_CXX11_HDR_SYSTEM_ERROR)
endif()

if(VSNRAY_ENABLE_COMMON)
add_subdirectory(common)
endif()

if(VSNRAY_ENABLE_EXAMPLES)
add_subdirectory(examples)
endif()

if(VSNRAY_ENABLE_VIEWER)
add_subdirectory(viewer)
endif()

add_subdirectory(visionaray)
