find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_RAYTRACINGINONEWEEKEND_SOURCES
    main.cpp
)

visionaray_add_executable(raytracinginoneweekend
    ${EX_RAYTRACINGINONEWEEKEND_SOURCES}
)
