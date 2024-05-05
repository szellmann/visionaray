# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(texture3d)
target_sources(texture3d PRIVATE main.cpp)
target_link_libraries(texture3d PUBLIC visionaray visionaray_common)
