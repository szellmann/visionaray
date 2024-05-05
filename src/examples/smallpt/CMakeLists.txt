# This file is distributed under the MIT license.
# See the LICENSE file for details.

add_executable(smallpt)
if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(smallpt PRIVATE main.cu)
else()
  target_sources(smallpt PRIVATE main.cpp)
endif()
target_link_libraries(smallpt PUBLIC visionaray visionaray_common)
