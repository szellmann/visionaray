find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_VOLUME_SOURCES
    ${CMD_LINE_DIR}/src/CmdLine.cpp
    ${CMD_LINE_DIR}/src/CmdLineUtil.cpp

    ${COMMON_DIR}/manip/arcball_manipulator.cpp
    ${COMMON_DIR}/manip/camera_manipulator.cpp
    ${COMMON_DIR}/manip/pan_manipulator.cpp
    ${COMMON_DIR}/manip/zoom_manipulator.cpp
    ${COMMON_DIR}/viewer_glut.cpp

    main.cpp
)

visionaray_add_executable(volume
    ${EX_VOLUME_SOURCES}
)
