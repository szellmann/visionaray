# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(volumetric_pathtracer)
target_sources(volumetric_pathtracer PRIVATE main.cpp)
target_link_libraries(volumetric_pathtracer PUBLIC visionaray visionaray_common)
