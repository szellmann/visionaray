# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# Add library target
#

add_library(visionaray)

target_sources(visionaray PRIVATE
    gl/compositing.cpp
    gl/handle.cpp
    gl/program.cpp
    gl/shader.cpp
    gl/util.cpp
    pixel_format.cpp
)

if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  set(VSNRAY_HAVE_CUDA 1)
  set(VSNRAY_HAVE_THRUST 1)
  target_sources(visionaray PRIVATE
      cuda/graphics_resource.cpp
      cuda/pixel_pack_buffer.cpp
      cuda/fill.cu
  )
  target_link_libraries(visionaray PUBLIC CUDA::cudart)
endif()

target_include_directories(visionaray PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Threads REQUIRED)
target_link_libraries(visionaray PUBLIC Threads::Threads)

# Graphics API

if (${VSNRAY_GRAPHICS_API} STREQUAL "GL")
  find_package(GLEW REQUIRED)
  find_package(OpenGL REQUIRED COMPONENTS OpenGL)
  target_link_libraries(visionaray PUBLIC GLEW::GLEW OpenGL::GL)
  set(VSNRAY_HAVE_GLEW 1)
  set(VSNRAY_HAVE_OPENGL 1)
elseif (${VSNRAY_GRAPHICS_API} STREQUAL "GLES")
  find_package(OpenGL REQUIRED COMPONENTS GLES2)
  target_link_libraries(visionaray PUBLIC OpenGL::GLES2)
  set(VSNRAY_HAVE_OPENGLES 1)
endif()

# TBB

if (VSNRAY_ENABLE_TBB)
  find_package(TBB REQUIRED)
  target_link_libraries(visionaray PUBLIC TBB::tbb)
  set(VSNRAY_HAVE_TBB 1)
endif()

#---------------------------------------------------------------------------------------------------
# Create config file


configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/visionaray/config.h
)


#--------------------------------------------------------------------------------------------------
# Install headers and library
#

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/visionaray ${PROJECT_SOURCE_DIR}/include/visionaray
    DESTINATION include
    COMPONENT headers
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hpp"
        PATTERN "*.hxx"
        PATTERN "*.inl"
        PATTERN "*.inc"
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/visionarayConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/visionarayConfig.cmake"
  INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

write_basic_package_version_file(
    "visionarayConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS visionaray
    EXPORT visionaray_Exports
    RUNTIME DESTINATION bin COMPONENT libraries
    LIBRARY DESTINATION lib COMPONENT libraries
    ARCHIVE DESTINATION lib COMPONENT libraries
)

install(EXPORT visionaray_Exports
    NAMESPACE visionaray::
    DESTINATION lib/cmake/visionaray
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/visionarayConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/visionarayConfigVersion.cmake
    DESTINATION lib/cmake/visionaray
)
