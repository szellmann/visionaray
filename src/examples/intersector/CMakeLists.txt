find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_INTERSECTOR_SOURCES
    main.cpp
    ${COMMON_DIR}/manip/arcball_manipulator.cpp
    ${COMMON_DIR}/manip/camera_manipulator.cpp
    ${COMMON_DIR}/manip/pan_manipulator.cpp
    ${COMMON_DIR}/manip/zoom_manipulator.cpp
)

visionaray_add_executable(intersector
    ${EX_INTERSECTOR_SOURCES}
)
