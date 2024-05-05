# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(volume)
target_sources(volume PRIVATE main.cpp)
target_link_libraries(volume PUBLIC visionaray visionaray_common)
