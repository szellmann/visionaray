# This file is distributed under the MIT license.
# See the LICENSE file for details.


#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost COMPONENTS filesystem iostreams system thread REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(JPEG)
find_package(OpenEXR)
find_package(OpenGL REQUIRED)
find_package(PNG)
find_package(PbrtParser)
find_package(Threads REQUIRED)
find_package(TIFF)
find_package(ZLIB)

visionaray_use_package(Boost)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenEXR)
visionaray_use_package(OpenGL)
visionaray_use_package(PNG)
visionaray_use_package(PbrtParser)
visionaray_use_package(Threads)
visionaray_use_package(TIFF)
visionaray_use_package(ZLIB)

# CUDA

if (VSNRAY_ENABLE_CUDA)
    find_package(CUDA)
    visionaray_use_package(CUDA)
endif()

# Cocoa

if (APPLE AND VSNRAY_ENABLE_COCOA)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(CORE_VIDEO_LIBRARY CoreVideo)
    visionaray_link_libraries(${COCOA_LIBRARY})
    visionaray_link_libraries(${CORE_VIDEO_LIBRARY})
endif()

# Ptex

if (VSNRAY_ENABLE_PTEX)
    find_package(Ptex)
    find_package(ZLIB)
    visionaray_use_package(Ptex)
    visionaray_use_package(ZLIB)
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

# SDL2

if (VSNRAY_ENABLE_SDL2)
    find_package(SDL2)
    visionaray_use_package(SDL2)
endif()

# TODO
set(3DCONNEXIONCLIENT_FOUND NOTFOUND)

if (APPLE AND VSNRAY_ENABLE_3DCONNEXIONCLIENT)
    include(CMakeFindFrameworks)

    CMAKE_FIND_FRAMEWORKS(3DconnexionClient)

    if (3DconnexionClient_FRAMEWORKS)
        set(__VSNRAY_LINK_LIBRARIES ${__VSNRAY_LINK_LIBRARIES} ${3DconnexionClient_FRAMEWORKS})
        set(__VSNRAY_USED_PACKAGES ${__VSNRAY_USED_PACKAGES} 3DCONNEXIONCLIENT)
        set(3DCONNEXIONCLIENT_FOUND FOUND)
    endif()
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

set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/imgui)
set(IMGUI_INCLUDE_DIR ${IMGUI_DIR})

set(RAPID_JSON_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/rapidjson)
set(RAPID_JSON_INCLUDE_DIR ${RAPID_JSON_DIR}/include)

set(TINYPLY_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/tinyply)
set(TINYPLY_INCLUDE_DIR ${TINYPLY_DIR}/source)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()


#---------------------------------------------------------------------------------------------------
# Create config file
#


foreach(p ${__VSNRAY_USED_PACKAGES})
    set(VSNRAY_COMMON_HAVE_${p} 1)
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
include_directories(${IMGUI_INCLUDE_DIR})
include_directories(${RAPID_JSON_INCLUDE_DIR})
include_directories(${TINYPLY_INCLUDE_DIR})

set(CMAKE_AUTOMOC ON)

set(COMMON_HEADERS

    ${CONFIG_DIR}/config.h

    async/connection.h
    async/connection_manager.h
    async/message.h

    input/cocoa.h
    input/glut.h
    input/key_event.h
    input/keyboard.h
    input/mouse.h
    input/mouse_event.h
    input/qt.h
    input/sdl2.h
    input/space_mouse.h

    manip/arcball.h
    manip/arcball_manipulator.h
    manip/camera_manipulator.h
    manip/model_manipulator.h
    manip/pan_manipulator.h
    manip/rotate_manipulator.h
    manip/translate_manipulator.h
    manip/zoom_manipulator.h

    # Scene graph

    sg/io.h

    blocking_queue.h
    cfile.h
    dds_image.h
    exr_image.h
    fbx_loader.h
    file_base.h
    hdr_image.h
    image.h
    image_base.h
    jpeg_image.h
    make_unique.h
    make_materials.h
    make_texture.h
    moana_loader.h
    model.h
    obj_grammar.h
    obj_loader.h
    pbrt_loader.h
    ply_loader.h
    png_image.h
    pnm_image.h
    ptex.h
    ptex.inl
    sg.h
    tga_image.h
    tiff_image.h
    timer.h
    viewer_base.h
    viewer_glut.h
    vsnray_loader.h

)

set(COMMON_SOURCES

    async/connection.cpp
    async/connection_manager.cpp
    async/message.cpp

    input/space_mouse.cpp

    manip/arcball.cpp
    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/first_person_manipulator.cpp
    manip/model_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/rotate_manipulator.cpp
    manip/translate_manipulator.cpp
    manip/zoom_manipulator.cpp

    dds_image.cpp
    exr_image.cpp
    fbx_loader.cpp
    file_base.cpp
    hdr_image.cpp
    image.cpp
    image_base.cpp
    inifile.cpp
    jpeg_image.cpp
    moana_loader.cpp
    model.cpp
    obj_grammar.cpp
    obj_loader.cpp
    pbrt_loader.cpp
    ply_loader.cpp
    png_image.cpp
    pnm_image.cpp
    sg.cpp
    tga_image.cpp
    tiff_image.cpp
    viewer_base.cpp
    viewer_glut.cpp
    vsnray_loader.cpp

)

if (APPLE AND VSNRAY_ENABLE_COCOA)
    set(COMMON_HEADERS ${COMMON_HEADERS} viewer_cocoa.h)
    set(COMMON_SOURCES ${COMMON_SOURCES} viewer_cocoa.mm)
endif()

if (VSNRAY_ENABLE_QT5)
    set(COMMON_HEADERS ${COMMON_HEADERS} viewer_qt.h)
    set(COMMON_SOURCES ${COMMON_SOURCES} viewer_qt.cpp)
endif()

if (VSNRAY_ENABLE_SDL2)
    set(COMMON_HEADERS ${COMMON_HEADERS} viewer_sdl2.h)
    set(COMMON_SOURCES ${COMMON_SOURCES} viewer_sdl2.cpp)
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


set(IMGUI_HEADERS
    ${IMGUI_INCLUDE_DIR}/imconfig.h
    ${IMGUI_INCLUDE_DIR}/imgui.h
    ${IMGUI_INCLUDE_DIR}/imgui_internal.h
    ${IMGUI_INCLUDE_DIR}/imstb_rectpack.h
    ${IMGUI_INCLUDE_DIR}/imstb_textedit.h
    ${IMGUI_INCLUDE_DIR}/imstb_truetype.h
)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)


set(TINYPLY_HEADERS
    ${TINYPLY_INCLUDE_DIR}/tinyply.h
)

set(TINYPLY_SOURCES
    ${TINYPLY_INCLUDE_DIR}/tinyply.cpp
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
    ${IMGUI_HEADERS}
    ${IMGUI_SOURCES}
    ${TINYPLY_HEADERS}
    ${TINYPLY_SOURCES}
)


#--------------------------------------------------------------------------------------------------
# Use Qt5 components - probably similar to visionaray_use_package(...)
#

if (VSNRAY_ENABLE_QT5)
    qt5_use_modules(visionaray_common Core Gui OpenGL Widgets)
endif()
