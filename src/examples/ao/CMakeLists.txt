# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(ao)
target_sources(ao PRIVATE main.cpp)
target_link_libraries(ao PUBLIC visionaray visionaray_common)
