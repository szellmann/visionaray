find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_AO_SOURCES
    main.cpp
)

visionaray_add_executable(ao
    ${EX_AO_SOURCES}
)
