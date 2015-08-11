find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_VOLUME_SOURCES
    main.cpp
)

visionaray_add_executable(volume
    ${EX_VOLUME_SOURCES}
)
