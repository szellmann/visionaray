find_package(Boost REQUIRED)
find_package(CUDA)
find_package(GLEW)
find_package(OpenGL)

visionaray_use_package(Boost)
visionaray_use_package(CUDA)
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/visionaray)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/visionaray)


set(VSNRAY_HEADERS

    #----------------------------------------------------------------------------------------------
    # Public headers
    #

    ${HEADER_DIR}/math/detail/aabb.inl

    ${HEADER_DIR}/camera.h

    #----------------------------------------------------------------------------------------------
    # Private headers
    #

    cuda/graphics_resource.h

    gl/handle.h
    gl/util.h

    util.h

)

set(VSNRAY_SOURCES

    cuda/graphics_resource.cpp

    gl/handle.cpp
    gl/util.cpp

    camera.cpp
    exception.cpp
    pixel_format.cpp
    pixel_unpack_buffer_rt.cpp
    render_target.cpp
    util.cpp

)


add_library(visionaray
    ${VSNRAY_HEADERS}
    ${VSNRAY_SOURCES}
)


