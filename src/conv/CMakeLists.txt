# This file is distributed under the MIT license.
# See the LICENSE file for details.

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)
set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()


#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)

visionaray_use_package(Boost)

# Ptex

if (VSNRAY_ENABLE_PTEX)
    find_package(Ptex)
    find_package(ZLIB)
    visionaray_use_package(Ptex)
    visionaray_use_package(ZLIB)
endif()


#--------------------------------------------------------------------------------------------------
#
#

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${__VSNRAY_CONFIG_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})

set(CONV_HEADERS
)

set(CONV_SOURCES
    conv.cpp
)


#--------------------------------------------------------------------------------------------------
# Add conv target
#

visionaray_add_executable(conv ${CONV_HEADERS} ${CONV_SOURCES})


#--------------------------------------------------------------------------------------------------
# Install conv
#

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/conv
    DESTINATION bin
    RENAME vsnray-conv
)
