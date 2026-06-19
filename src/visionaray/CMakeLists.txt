# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# Add library target
#

add_library(visionaray INTERFACE)

if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  target_link_libraries(visionaray INTERFACE CUDA::cudart)
endif()

if (VSNRAY_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(rocthrust REQUIRED)
  find_package(hipcub REQUIRED)
  # HIP libraries are available but NOT linked to the INTERFACE target
  # Targets with HIP sources should link hip::host/device, roc::rocthrust, hip::hipcub
  # directly to avoid leaking HIP compiler options to CXX-only targets
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
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
endif()


#--------------------------------------------------------------------------------------------------
# Install
#

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/visionaray ${PROJECT_SOURCE_DIR}/include/visionaray
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
)

install(EXPORT visionaray_Exports
    NAMESPACE visionaray::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/visionarayConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/visionarayConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
