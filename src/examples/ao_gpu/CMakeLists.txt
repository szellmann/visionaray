find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)

visionaray_use_package(CUDA)
visionaray_use_package(GLUT)

visionaray_cuda_compile(EX_AO_GPU_SOURCES
    main.cu
)

visionaray_add_executable(ao_gpu
    ${EX_AO_GPU_SOURCES}
)
