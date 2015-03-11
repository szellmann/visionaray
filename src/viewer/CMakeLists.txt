find_package(Boost REQUIRED COMPONENTS chrono filesystem iostreams system thread)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(JPEG)
find_package(OpenGL REQUIRED)

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenGL)

visionaray_link_libraries(visionaray)

set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(VIEWER_HEADERS

    ${COMMON_DIR}/call_kernel.h
    ${COMMON_DIR}/render_bvh.h
    ${COMMON_DIR}/timer.h
    ${COMMON_DIR}/util.h

    input/exception.h
    input/keyboard.h
    input/mouse.h

    manip/arcball_manipulator.h
    manip/camera_manipulator.h
    manip/pan_manipulator.h
    manip/zoom_manipulator.h

    default_scenes.h
    image.h
    jpeg_image.h
    obj_loader.h
)

set(VIEWER_SOURCES

    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/zoom_manipulator.cpp

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

visionaray_add_executable(viewer
    ${VIEWER_HEADERS}
    ${VIEWER_SOURCES}
    ${VIEWER_CUDA_SOURCES}
)
