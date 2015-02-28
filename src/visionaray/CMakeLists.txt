find_package(Boost REQUIRED)
find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

visionaray_use_package(Boost)
visionaray_use_package(CUDA)
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/visionaray)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/visionaray)


set(VSNRAY_HEADERS

    #----------------------------------------------------------------------------------------------
    # Public headers
    #

    # Details - subject to frequent change!

    ${HEADER_DIR}/detail/aligned_vector.h
    ${HEADER_DIR}/detail/allocator.h
    ${HEADER_DIR}/detail/bvh_build_binned.inl
    ${HEADER_DIR}/detail/bvh_intersect.inl
    ${HEADER_DIR}/detail/color_conversion.h
    ${HEADER_DIR}/detail/compiler.h
    ${HEADER_DIR}/detail/cuda_sched.h
    ${HEADER_DIR}/detail/cuda_sched.inl
    ${HEADER_DIR}/detail/macros.h
    ${HEADER_DIR}/detail/material.inl
    ${HEADER_DIR}/detail/pathtracing.inl
    ${HEADER_DIR}/detail/platform.h
    ${HEADER_DIR}/detail/sched_common.h
    ${HEADER_DIR}/detail/semaphore.h
    ${HEADER_DIR}/detail/simple.inl
    ${HEADER_DIR}/detail/simple_sched.h
    ${HEADER_DIR}/detail/simple_sched.inl
    ${HEADER_DIR}/detail/stack.h
    ${HEADER_DIR}/detail/surface.inl
    ${HEADER_DIR}/detail/tags.h
    ${HEADER_DIR}/detail/tiled_sched.h
    ${HEADER_DIR}/detail/tiled_sched.inl
    ${HEADER_DIR}/detail/traverse.h
    ${HEADER_DIR}/detail/traverse_linear.inl
    ${HEADER_DIR}/detail/whitted.inl

    # Math

    ${HEADER_DIR}/math/detail/simd/matrix4.inl
    ${HEADER_DIR}/math/detail/aabb.inl
    ${HEADER_DIR}/math/detail/math.h
    ${HEADER_DIR}/math/detail/matrix4.inl
    ${HEADER_DIR}/math/detail/plane3.inl
    ${HEADER_DIR}/math/detail/quat.inl
    ${HEADER_DIR}/math/detail/rectangle.inl
    ${HEADER_DIR}/math/detail/vector.inl
    ${HEADER_DIR}/math/detail/vector2.inl
    ${HEADER_DIR}/math/detail/vector3.inl
    ${HEADER_DIR}/math/detail/vector4.inl
    ${HEADER_DIR}/math/simd/avx.h
    ${HEADER_DIR}/math/simd/forward.h
    ${HEADER_DIR}/math/simd/intrinsics.h
    ${HEADER_DIR}/math/simd/matrix.h
    ${HEADER_DIR}/math/simd/simd.h
    ${HEADER_DIR}/math/simd/sse.h
    ${HEADER_DIR}/math/simd/trans.h
    ${HEADER_DIR}/math/aabb.h
    ${HEADER_DIR}/math/axis.h
    ${HEADER_DIR}/math/config.h
    ${HEADER_DIR}/math/forward.h
    ${HEADER_DIR}/math/intersect.h
    ${HEADER_DIR}/math/math.h
    ${HEADER_DIR}/math/matrix.h
    ${HEADER_DIR}/math/output.h
    ${HEADER_DIR}/math/primitive.h
    ${HEADER_DIR}/math/plane.h
    ${HEADER_DIR}/math/quat.h
    ${HEADER_DIR}/math/ray.h
    ${HEADER_DIR}/math/rectangle.h
    ${HEADER_DIR}/math/serialization.h
    ${HEADER_DIR}/math/sphere.h
    ${HEADER_DIR}/math/triangle.h
    ${HEADER_DIR}/math/vector.h

    # Texture access

    ${HEADER_DIR}/texture/detail/prefilter.h
    ${HEADER_DIR}/texture/detail/sampler1d.h
    ${HEADER_DIR}/texture/detail/sampler2d.h
    ${HEADER_DIR}/texture/detail/sampler3d.h
    ${HEADER_DIR}/texture/detail/sampler_common.h
    ${HEADER_DIR}/texture/detail/texture1d.h
    ${HEADER_DIR}/texture/detail/texture2d.h
    ${HEADER_DIR}/texture/detail/texture3d.h
    ${HEADER_DIR}/texture/detail/texture_common.h
    ${HEADER_DIR}/texture/forward.h
    ${HEADER_DIR}/texture/texture.h

    # General library headers

    ${HEADER_DIR}/brdf.h
    ${HEADER_DIR}/bvh
    ${HEADER_DIR}/camera.h
    ${HEADER_DIR}/exception.h
    ${HEADER_DIR}/generic_prim.h
    ${HEADER_DIR}/gpu_buffer_rt.h
    ${HEADER_DIR}/kernels.h
    ${HEADER_DIR}/material.h
    ${HEADER_DIR}/mc.h
    ${HEADER_DIR}/norm.h
    ${HEADER_DIR}/pixel_format.h
    ${HEADER_DIR}/pixel_traits.h
    ${HEADER_DIR}/pixel_unpack_buffer_rt.h
    ${HEADER_DIR}/point_light.h
    ${HEADER_DIR}/render_target.h
    ${HEADER_DIR}/scheduler.h
    ${HEADER_DIR}/shade_record.h
    ${HEADER_DIR}/surface.h
    ${HEADER_DIR}/tags.h

    #----------------------------------------------------------------------------------------------
    # Private headers
    #

    cuda/graphics_resource.h

    gl/handle.h
    gl/util.h

    util.h

)

set(VSNRAY_SOURCES

    cuda/graphics_resource.cpp

    gl/handle.cpp
    gl/util.cpp

    camera.cpp
    exception.cpp
    pixel_format.cpp
    pixel_unpack_buffer_rt.cpp
    render_target.cpp
    util.cpp

)


add_library(visionaray
    ${VSNRAY_HEADERS}
    ${VSNRAY_SOURCES}
)


