# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# Add library target
#

add_library(visionaray INTERFACE)

if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  set(VSNRAY_HAVE_CUDA 1)
  set(VSNRAY_HAVE_THRUST 1)
  target_link_libraries(visionaray INTERFACE CUDA::cudart)
endif()

target_include_directories(visionaray INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Threads REQUIRED)
target_link_libraries(visionaray INTERFACE Threads::Threads)

# TBB

if (VSNRAY_ENABLE_TBB)
  find_package(TBB REQUIRED)
  target_link_libraries(visionaray INTERFACE TBB::tbb)
  set(VSNRAY_HAVE_TBB 1)
endif()


#--------------------------------------------------------------------------------------------------
# Install
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
