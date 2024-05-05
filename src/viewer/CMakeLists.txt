# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# Add viewer target
#

add_executable(viewer)

target_sources(viewer PRIVATE
  host_device_rt.cpp
  render_generic_material.cpp
  render_instances.cpp
  render_instances_ptex.cpp
  render_plastic.cpp
)

if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(viewer PRIVATE
    render_generic_material.cu
    render_instances.cu
    render_plastic.cu
    viewer.cu
  )
else()
  target_sources(viewer PRIVATE viewer.cpp)
endif()


target_link_libraries(viewer PUBLIC visionaray)
target_link_libraries(viewer PUBLIC visionaray_common)

#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(GLUT REQUIRED)

#--------------------------------------------------------------------------------------------------
# Install viewer
#

install(TARGETS viewer
    DESTINATION bin
    RENAME vsnray-viewer
)
