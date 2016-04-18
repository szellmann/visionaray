set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{COVISEDIR}/cmake")
find_package(COVER)
find_package(COVISE)
if (NOT COVER_FOUND)
   message("COVER not found - not building COVER plugins")
   return()
endif()

find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()
find_package(Visionaray REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    ${Boost_LIBRARIES}
    ${GLEW_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${VISIONARAY_LIBRARY}
)

if (NOT APPLE AND NOT WIN32)
include_directories(${PTHREAD_INCLUDE_DIR})
set(EXTRA_LIBS
    ${EXTRA_LIBS}
    ${PTHREAD_LIBRARY}
)
endif()

set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

set(HEADERS
    ${COMMON_DIR}/call_kernel.h
    ${COMMON_DIR}/render_bvh.h

    kernels/bvh_costs_kernel.h
    kernels/normals_kernel.h
    kernels/tex_coords_kernel.h
    drawable.h
    state.h
    visionaray_plugin.h
)

set(SOURCES
    ${COMMON_DIR}/util.cpp

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
