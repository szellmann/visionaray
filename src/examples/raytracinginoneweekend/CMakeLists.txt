# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(raytracinginoneweekend)
target_sources(raytracinginoneweekend PRIVATE main.cpp)
target_link_libraries(raytracinginoneweekend PUBLIC visionaray visionaray_common)
