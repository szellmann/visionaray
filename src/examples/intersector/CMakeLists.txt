find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_INTERSECTOR_SOURCES
    ${COMMON_DIR}/manip/arcball_manipulator.cpp
    ${COMMON_DIR}/manip/camera_manipulator.cpp
    ${COMMON_DIR}/manip/pan_manipulator.cpp
    ${COMMON_DIR}/manip/zoom_manipulator.cpp
    ${COMMON_DIR}/viewer_glut.cpp

    main.cpp
)

visionaray_add_executable(intersector
    ${EX_INTERSECTOR_SOURCES}
)
