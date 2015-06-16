if (NOT VSNRAY_ENABLE_COVER)
    return()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{COVISEDIR}/cmake")
find_package(COVER)
find_package(COVISE)
if (NOT COVER_FOUND)
   message("COVER not found - not building COVER plugins")
   return()
endif()

find_package(Boost REQUIRED COMPONENTS chrono filesystem iostreams system thread)
find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()
find_package(Visionaray REQUIRED)

visionaray_use_package(COVER)
visionaray_use_package(COVISE)
visionaray_use_package(CUDA)
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)
if (NOT APPLE AND NOT WIN32)
visionaray_use_package(PTHREAD)
endif()
visionaray_use_package(Visionaray)

include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXTRA_LIBS ${EXTRA_LIBS} ${GLEW_LIBRARY} ${VISIONARAY_LIBRARY})

set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

set(HEADERS
    ${COMMON_DIR}/call_kernel.h
    ${COMMON_DIR}/render_bvh.h

    drawable.h
    state.h
    visionaray_plugin.h
)

set(SOURCES
    visionaray_plugin.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
    visionaray_cuda_compile(CUDA_SOURCES
        drawable.cu
    )
else()
    set(SOURCES
        ${SOURCES}
        drawable.cpp
    )
endif()


cover_add_plugin(Visionaray ${HEADERS} ${SOURCES} ${CUDA_SOURCES})

target_link_libraries(Visionaray ${EXTRA_LIBS})
