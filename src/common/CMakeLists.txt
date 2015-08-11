find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)
if (NOT APPLE AND NOT WIN32)
visionaray_use_package(PTHREAD)
endif()

visionaray_link_libraries(visionaray)

set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/common/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMD_LINE_INCLUDE_DIR})

set(COMMON_HEADERS

    input/exception.h
    input/keyboard.h
    input/mouse.h

    manip/arcball_manipulator.h
    manip/camera_manipulator.h
    manip/pan_manipulator.h
    manip/zoom_manipulator.h

    call_kernel.h
    image.h
    jpeg_image.h
    obj_loader.h
    render_bvh.h
    timer.h
    util.h
    viewer_glut.h
)

set(COMMON_SOURCES

    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/zoom_manipulator.cpp

    jpeg_image.cpp
    obj_loader.cpp
    util.cpp
    viewer_glut.cpp

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
