# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(multi_volume)
if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(multi_volume PRIVATE main.cu)
else()
  target_sources(multi_volume PRIVATE main.cpp)
endif()
target_link_libraries(multi_volume PUBLIC visionaray visionaray_common)
