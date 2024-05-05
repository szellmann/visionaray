# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(generic_primitive)
target_sources(generic_primitive PRIVATE main.cpp)
target_link_libraries(generic_primitive PUBLIC visionaray visionaray_common)
