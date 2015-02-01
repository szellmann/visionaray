find_package(Boost REQUIRED COMPONENTS chrono filesystem system thread)
find_package(CUDA)
find_package(GLEW)
find_package(GLUT)
find_package(JPEG)
find_package(OpenGL)

visionaray_use_package(Boost)
visionaray_use_package(CUDA)
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(JPEG)
visionaray_use_package(OpenGL)

visionaray_link_libraries(visionaray)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(VIEWER_HEADERS
    jpeg_image.h
    obj_loader.h
    timer.h
)

set(VIEWER_SOURCES

    manip/arcball_manipulator.cpp
    manip/camera_manipulator.cpp
    manip/pan_manipulator.cpp
    manip/zoom_manipulator.cpp

    jpeg_image.cpp
    obj_loader.cpp

)

if(CUDA_FOUND)
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

