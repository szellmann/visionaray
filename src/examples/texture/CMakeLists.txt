find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_TEXTURE_SOURCES
    main.cpp
)

visionaray_add_executable(texture
    ${EX_TEXTURE_SOURCES}
)
