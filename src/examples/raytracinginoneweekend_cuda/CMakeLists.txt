find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_RAYTRACINGINONEWEEKEND_CUDA_SOURCES
    main.cu
)

visionaray_add_cuda_executable(raytracinginoneweekend_cuda
    ${EX_RAYTRACINGINONEWEEKEND_CUDA_SOURCES}
)
