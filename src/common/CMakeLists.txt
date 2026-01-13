# This file is distributed under the MIT license.
# See the LICENSE file for details.

#--------------------------------------------------------------------------------------------------
# Add library target
#

add_library(visionaray_common)

target_sources(visionaray_common PRIVATE
  ../3rdparty/CmdLine/src/CmdLine.cpp
  ../3rdparty/CmdLine/src/CmdLineUtil.cpp
  ../3rdparty/imgui/imgui.cpp
  ../3rdparty/imgui/imgui_draw.cpp
  ../3rdparty/imgui/imgui_tables.cpp
  ../3rdparty/imgui/imgui_widgets.cpp
  ../3rdparty/tinyply/source/tinyply.cpp
  cuda/graphics_resource.cpp
  cuda/pixel_pack_buffer.cpp
  gl/compositing.cpp
  gl/handle.cpp
  gl/program.cpp
  gl/shader.cpp
  gl/util.cpp
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
  bvh_outline_renderer.cpp
  dds_image.cpp
  exr_image.cpp
  fbx_loader.cpp
  file_base.cpp
  gl_debug_callback.cpp
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
  pixel_format.cpp
  ply_loader.cpp
  png_image.cpp
  pnm_image.cpp
  sg.cpp
  tga_image.cpp
  tiff_image.cpp
  viewer_base.cpp
  vsnray_loader.cpp
)

if (VSNRAY_ENABLE_CUDA)
  enable_language(CUDA)
  set(VSNRAY_COMMON_HAVE_CUDA 1)
  set(VSNRAY_COMMON_HAVE_THRUST 1)
endif()

target_include_directories(visionaray_common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/3rdparty/CmdLine/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/3rdparty/imgui>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/3rdparty/rapidjson/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/3rdparty/tinyply/source>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/common/imgui>
)

target_link_libraries(visionaray_common PUBLIC visionaray)

# Enable multi-processor compilation for MSVC
if(MSVC)
  target_compile_options(visionaray_common PRIVATE /MP)
endif()

#--------------------------------------------------------------------------------------------------
# External libraries
#

find_package(Boost CONFIG COMPONENTS filesystem iostreams REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

find_package(GLUT)
find_package(JPEG)
find_package(OpenEXR)
find_package(PNG)
find_package(Ptex)
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5Widgets)
find_package(SDL2)
find_package(TBB)
find_package(TIFF)
find_package(ZLIB)

target_link_libraries(visionaray_common PUBLIC
    Boost::disable_autolinking
    Boost::filesystem
    Boost::iostreams
    GLEW::GLEW
    OpenGL::GL
    Threads::Threads
)
set(VSNRAY_COMMON_HAVE_BOOST 1)
set(VSNRAY_COMMON_HAVE_GLEW 1)
set(VSNRAY_COMMON_HAVE_OPENGL 1)
set(VSNRAY_COMMON_HAVE_THREADS 1)

if (TARGET GLUT::GLUT)
  target_sources(visionaray_common PRIVATE viewer_glut.cpp)
  target_link_libraries(visionaray_common PUBLIC GLUT::GLUT)
  set(VSNRAY_COMMON_HAVE_GLUT 1)
endif()

if (TARGET OpenEXR::OpenEXR)
  target_link_libraries(visionaray_common PUBLIC OpenEXR::OpenEXR)
  set(VSNRAY_COMMON_HAVE_OPENEXR 1)
endif()

if (TARGET PNG::PNG)
  target_link_libraries(visionaray_common PUBLIC PNG::PNG)
  set(VSNRAY_COMMON_HAVE_PNG 1)
endif()

if (VSNRAY_ENABLE_PTEX AND TARGET Ptex::Ptex)
  target_link_libraries(visionaray_common PUBLIC Ptex::Ptex)
  set(VSNRAY_COMMON_HAVE_PTEX 1)
endif()

if (VSNRAY_ENABLE_QT AND TARGET Qt5::Core)
  set(CMAKE_AUTOMOC ON)
  target_sources(visionaray_common PRIVATE viewer_qt.cpp)
  target_link_libraries(visionaray_common PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::OpenGL
    Qt5::Widgets
  )
  set(VSNRAY_COMMON_HAVE_QT5 1)
endif()

if (VSNRAY_ENABLE_SDL2 AND TARGET SDL2::SDL)
  target_sources(visionaray_common PRIVATE viewer_sdl2.cpp)
  target_link_libraries(visionaray_common PUBLIC SDL2::SDL)
  set(VSNRAY_COMMON_HAVE_SDL2 1)
endif()

if (VSNRAY_ENABLE_TBB AND TARGET TBB::TBB)
  target_link_libraries(visionaray_common PUBLIC TBB::TBB)
  set(VSNRAY_COMMON_HAVE_TBB 1)
endif()

if (TARGET TIFF::TIFF)
  target_link_libraries(visionaray_common PUBLIC TIFF::TIFF)
  set(VSNRAY_COMMON_HAVE_TIFF 1)
endif()

if (TARGET ZLIB::ZLIB)
  target_link_libraries(visionaray_common PUBLIC ZLIB::ZLIB)
  set(VSNRAY_COMMON_HAVE_ZLIB 1)
endif()

if (APPLE AND VSNRAY_ENABLE_COCOA)
  find_library(COCOA_LIBRARY Cocoa)
  find_library(CORE_VIDEO_LIBRARY CoreVideo)
  target_sources(visionaray_common PRIVATE viewer_cocoa.mm)
  target_link_libraries(visionaray_common PUBLIC ${COCOA_LIBRARY})
  target_link_libraries(visionaray_common PUBLIC ${CORE_VIDEO_LIBRARY})
endif()

if (VSNRAY_ENABLE_PBRT_PARSER)
  target_link_libraries(visionaray_common PUBLIC pbrtParser)
  set(VSNRAY_COMMON_HAVE_PBRT_PARSER 1)
endif()

if (APPLE AND VSNRAY_ENABLE_3DCONNEXIONCLIENT)
  include(CMakeFindFrameworks)

  CMAKE_FIND_FRAMEWORKS(3DconnexionClient)

  target_link_libraries(visionaray_common PUBLIC ${3DconnexionClient_FRAMEWORKS})
  set(VSNRAY_COMMON_HAVE_3DCONNEXIONCLIENT 1)
endif()


#---------------------------------------------------------------------------------------------------
# Create config file
#


configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)


#--------------------------------------------------------------------------------------------------
# Install headers and library
#

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
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

option(VSNRAY_COMMON_INSTALL_IMGUI "Install ImGUI headers" ON)
set(IMGUI_DIR ../3rdparty/imgui)
if (VSNRAY_COMMON_INSTALL_IMGUI)
  install(DIRECTORY ${IMGUI_DIR}
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
      COMPONENT headers
      FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hh"
          PATTERN "*.hpp"
          PATTERN "*.hxx"
          PATTERN "*.inl"
          PATTERN "*.inc"
  )
endif()

install(TARGETS visionaray_common
    EXPORT visionaray_Exports
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
)
