# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(phantom)
target_sources(phantom PRIVATE main.cpp)
target_link_libraries(phantom PUBLIC visionaray visionaray_common)
