# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(intersector)
target_sources(intersector PRIVATE main.cpp)
target_link_libraries(intersector PUBLIC visionaray visionaray_common)
