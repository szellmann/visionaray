
# TODO!!
set(VSNRAY_LIBRARY "/home/zellmans/visionaray/build/src/visionaray/libvisionaray.so")

set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVER)
if (NOT COVER_FOUND)
   message("COVER not found - not building COVER plugins")
   return()
endif()

covise_create_using()

find_package(Boost REQUIRED COMPONENTS chrono filesystem iostreams system thread)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

set(EXTRA_LIBS ${EXTRA_LIBS} ${GLEW_LIBRARY} ${VSNRAY_LIBRARY})


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(HEADERS
    visionaray_plugin.h
)

set(SOURCES
    visionaray_plugin.cpp
)

cover_add_plugin(Visionaray ${HEADERS} ${SOURCES})

target_link_libraries(Visionaray ${EXTRA_LIBS})
