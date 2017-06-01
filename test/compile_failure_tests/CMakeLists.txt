# This file is distributed under the MIT license.
# See the LICENSE file for details.

# Visionaray include dir
include_directories(${PROJECT_SOURCE_DIR}/include)
# Also add this so we can include common headers
include_directories(${PROJECT_SOURCE_DIR}/src)
# Find config headers
include_directories(${__VSNRAY_CONFIG_DIR})

add_subdirectory(algo)
add_subdirectory(packing)
add_subdirectory(simd)
