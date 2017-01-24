#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(JPEG)
find_package(OpenGL REQUIRED)
find_package(PNG)
find_package(Threads REQUIRED)
find_package(TIFF)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenGL)
visionaray_use_package(PNG)
visionaray_use_package(Threads)
visionaray_use_package(TIFF)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

# Qt5

if (VSNRAY_ENABLE_QT5)
    find_package(Qt5Core)
    find_package(Qt5Gui)
    find_package(Qt5OpenGL)
    find_package(Qt5Widgets)
    visionaray_use_package(Qt5Core)
    visionaray_use_package(Qt5Gui)
    visionaray_use_package(Qt5OpenGL)
    visionaray_use_package(Qt5Widgets)
endif()


#--------------------------------------------------------------------------------------------------
#
#

visionaray_link_libraries(visionaray)

if (WIN32)
    add_definitions(-DBOOST_ALL_NO_LIB)
    set(Boost_USE_STATIC_LIBS ON)
endif()

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()


#---------------------------------------------------------------------------------------------------
# Create config file
#


foreach(p ${__VSNRAY_USED_PACKAGES})
    set(VSNRAY_HAVE_${p} 1)
endforeach()

set(CONFIG_DIR ${__VSNRAY_COMMON_CONFIG_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CONFIG_DIR}/config.h)

include_directories(${__VSNRAY_CONFIG_DIR})


#---------------------------------------------------------------------------------------------------
# Visionaray common library
#

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMD_LINE_INCLUDE_DIR})

set(CMAKE_AUTOMOC ON)

set(COMMON_HEADERS

    ${CONFIG_DIR}/config.h

    input/exception.h
    input/glut.h
    input/keyboard.h
    input/mouse.h
    input/qt.h
    input/sdl2.h

    manip/arcball.h
    manip/arcball_manipulator.h
    manip/camera_manipulator.h
    manip/model_manipulator.h
    manip/pan_manipulator.h
    manip/rotate_manipulator.h
    manip/translate_manipulator.h
    manip/zoom_manipulator.h

    call_kernel.h
    cfile.h
    hdr_image.h
    image.h
    image_base.h
    jpeg_image.h
    make_unique.h
    model.h
    obj_grammar.h
    obj_loader.h
    png_image.h
    pnm_image.h
    tga_image.h
    tiff_image.h
    timer.h
    viewer_base.h
    viewer_glut.h
)

set(COMMON_SOURCES

    manip/arcball.cpp
    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/model_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/rotate_manipulator.cpp
    manip/translate_manipulator.cpp
    manip/zoom_manipulator.cpp

    hdr_image.cpp
    image.cpp
    image_base.cpp
    jpeg_image.cpp
    obj_grammar.cpp
    obj_loader.cpp
    png_image.cpp
    pnm_image.cpp
    tga_image.cpp
    tiff_image.cpp
    viewer_base.cpp
    viewer_glut.cpp

)

if (VSNRAY_ENABLE_QT5)

set(COMMON_HEADERS
    ${COMMON_HEADERS}
    viewer_qt.h
)

set(COMMON_SOURCES
    ${COMMON_SOURCES}
    viewer_qt.cpp

)

endif()

set(CMD_LINE_HEADERS

    ${CMD_LINE_INCLUDE_DIR}/Support/CmdLine.h
    ${CMD_LINE_INCLUDE_DIR}/Support/CmdLineUtil.h
    ${CMD_LINE_INCLUDE_DIR}/Support/StringRef.h
    ${CMD_LINE_INCLUDE_DIR}/Support/StringRefStream.h
    ${CMD_LINE_INCLUDE_DIR}/Support/StringSplit.h

)

set(CMD_LINE_SOURCES

    ${CMD_LINE_DIR}/src/CmdLine.cpp
    ${CMD_LINE_DIR}/src/CmdLineUtil.cpp

)


#--------------------------------------------------------------------------------------------------
# Add library target
#

visionaray_add_library(visionaray_common
    STATIC
    ${COMMON_HEADERS}
    ${COMMON_SOURCES}
    ${CMD_LINE_HEADERS}
    ${CMD_LINE_SOURCES}
)


#--------------------------------------------------------------------------------------------------
# Use Qt5 components - probably similar to visionaray_use_package(...)
#

if (VSNRAY_ENABLE_QT5)
qt5_use_modules(visionaray_common Core Gui OpenGL Widgets)
endif()
