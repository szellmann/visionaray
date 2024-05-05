# This file is distributed under the MIT license.
# See the LICENSE file for details.

enable_language(CUDA)
add_executable(raytracinginoneweekend_cuda)
target_sources(raytracinginoneweekend_cuda PRIVATE main.cu)
target_link_libraries(raytracinginoneweekend_cuda PUBLIC visionaray visionaray_common)
