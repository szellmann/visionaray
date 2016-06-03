find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_MULTI_HIT_SOURCES
    main.cpp
)

visionaray_add_executable(multi_hit
    ${EX_MULTI_HIT_SOURCES}
)
