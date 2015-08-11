find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_INTERSECTOR_SOURCES
    main.cpp
)

visionaray_add_executable(intersector
    ${EX_INTERSECTOR_SOURCES}
)
