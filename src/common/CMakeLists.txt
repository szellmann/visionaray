find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(JPEG)
find_package(OpenGL REQUIRED)
find_package(PNG)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5Widgets)

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenGL)
visionaray_use_package(PNG)
if (NOT APPLE AND NOT WIN32)
visionaray_use_package(PTHREAD)
endif()
visionaray_use_package(Qt5Core)
visionaray_use_package(Qt5Gui)
visionaray_use_package(Qt5OpenGL)
visionaray_use_package(Qt5Widgets)

visionaray_link_libraries(visionaray)

if (WIN32)
    add_definitions(-DBOOST_ALL_NO_LIB)
    set(Boost_USE_STATIC_LIBS ON)
endif()

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/common/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMD_LINE_INCLUDE_DIR})

set(CMAKE_AUTOMOC ON)

set(COMMON_HEADERS

    input/exception.h
    input/keyboard.h
    input/mouse.h

    manip/arcball_manipulator.h
    manip/camera_manipulator.h
    manip/pan_manipulator.h
    manip/zoom_manipulator.h

    call_kernel.h
    cfile.h
    image.h
    jpeg_image.h
    make_unique.h
    obj_loader.h
    png_image.h
    render_bvh.h
    tga_image.h
    timer.h
    util.h
    viewer_base.h
    viewer_glut.h
    viewer_qt.h
)

set(COMMON_SOURCES

    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/zoom_manipulator.cpp

    jpeg_image.cpp
    obj_loader.cpp
    png_image.cpp
    tga_image.cpp
    util.cpp
    viewer_base.cpp
    viewer_glut.cpp
    viewer_qt.cpp

)

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

qt5_use_modules(visionaray_common Core Gui OpenGL Widgets)
