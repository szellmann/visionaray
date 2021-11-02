find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_OPENGL_INTEROP_SOURCES
    main.cu
)

visionaray_add_cuda_executable(opengl_interop
    ${EX_OPENGL_INTEROP_SOURCES}
)

target_compile_definitions(opengl_interop PUBLIC APPDIR="${CMAKE_CURRENT_SOURCE_DIR}")
