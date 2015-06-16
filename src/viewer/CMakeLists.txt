set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)
set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/viewer/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()


find_package(Boost REQUIRED COMPONENTS chrono filesystem iostreams system thread)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(JPEG)
find_package(OpenGL REQUIRED)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenGL)
if (NOT APPLE AND NOT WIN32)
visionaray_use_package(PTHREAD)
endif()

visionaray_link_libraries(visionaray)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMD_LINE_INCLUDE_DIR})

set(VIEWER_HEADERS

    ${COMMON_DIR}/input/exception.h
    ${COMMON_DIR}/input/keyboard.h
    ${COMMON_DIR}/input/mouse.h

    ${COMMON_DIR}/manip/arcball_manipulator.h
    ${COMMON_DIR}/manip/camera_manipulator.h
    ${COMMON_DIR}/manip/pan_manipulator.h
    ${COMMON_DIR}/manip/zoom_manipulator.h

    ${COMMON_DIR}/call_kernel.h
    ${COMMON_DIR}/render_bvh.h
    ${COMMON_DIR}/timer.h
    ${COMMON_DIR}/util.h

    default_scenes.h
    image.h
    jpeg_image.h
    obj_loader.h
)

set(VIEWER_SOURCES

    ${COMMON_DIR}/manip/arcball_manipulator.cpp
    ${COMMON_DIR}/manip/camera_manipulator.cpp
    ${COMMON_DIR}/manip/pan_manipulator.cpp
    ${COMMON_DIR}/manip/zoom_manipulator.cpp

    jpeg_image.cpp
    obj_loader.cpp

)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(VIEWER_CUDA_SOURCES
        viewer.cu
    )
else()
    set(VIEWER_SOURCES
        ${VIEWER_SOURCES}
        viewer.cpp
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
# Add viewer target
#

visionaray_add_executable(viewer
    ${VIEWER_HEADERS}
    ${VIEWER_SOURCES}
    ${VIEWER_CUDA_SOURCES}
    ${CMD_LINE_HEADERS}
    ${CMD_LINE_SOURCES}
)


#--------------------------------------------------------------------------------------------------
# Install viewer
#

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/viewer
    DESTINATION bin
    RENAME vsnray-viewer
)
