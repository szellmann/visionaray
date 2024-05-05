# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(juggler)
target_sources(juggler PRIVATE main.cpp)
target_link_libraries(juggler PUBLIC visionaray visionaray_common)
