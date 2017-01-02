find_package(Boost REQUIRED)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(OpenGL)
visionaray_use_package(Threads)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/visionaray)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/visionaray)


set(VSNRAY_HEADERS

    #----------------------------------------------------------------------------------------------
    # Public headers
    #

    # Cuda

    ${HEADER_DIR}/cuda/detail/pixel_pack_buffer.inl
    ${HEADER_DIR}/cuda/array.h
    ${HEADER_DIR}/cuda/cast.h
    ${HEADER_DIR}/cuda/graphics_resource.h
    ${HEADER_DIR}/cuda/pitch2d.h
    ${HEADER_DIR}/cuda/pixel_pack_buffer.h
    ${HEADER_DIR}/cuda/util.h

    # Details - subject to frequent change!

    ${HEADER_DIR}/detail/bvh/build.inl
    ${HEADER_DIR}/detail/bvh/get_bounds.inl
    ${HEADER_DIR}/detail/bvh/get_color.h
    ${HEADER_DIR}/detail/bvh/get_normal.h
    ${HEADER_DIR}/detail/bvh/get_tex_coord.h
    ${HEADER_DIR}/detail/bvh/hit_record.h
    ${HEADER_DIR}/detail/bvh/intersect.inl
    ${HEADER_DIR}/detail/bvh/prim_traits.h
    ${HEADER_DIR}/detail/bvh/sah.h
    ${HEADER_DIR}/detail/bvh/statistics.h
    ${HEADER_DIR}/detail/bvh/traverse.h
    ${HEADER_DIR}/detail/material/emissive.inl
    ${HEADER_DIR}/detail/material/matte.inl
    ${HEADER_DIR}/detail/material/mirror.inl
    ${HEADER_DIR}/detail/material/plastic.inl
    ${HEADER_DIR}/detail/spd/blackbody.h
    ${HEADER_DIR}/detail/spd/d65.h
    ${HEADER_DIR}/detail/algorithm.h
    ${HEADER_DIR}/detail/allocator.h
    ${HEADER_DIR}/detail/area_light.inl
    ${HEADER_DIR}/detail/array.inl
    ${HEADER_DIR}/detail/camera.inl
    ${HEADER_DIR}/detail/color_conversion.h
    ${HEADER_DIR}/detail/compiler.h
    ${HEADER_DIR}/detail/cpu_buffer_rt.inl
    ${HEADER_DIR}/detail/cuda_sched.h
    ${HEADER_DIR}/detail/cuda_sched.inl
    ${HEADER_DIR}/detail/exit_traversal.h
    ${HEADER_DIR}/detail/generic_material.inl
    ${HEADER_DIR}/detail/generic_primitive.inl
    ${HEADER_DIR}/detail/generic_primitive_get_normal.h
    ${HEADER_DIR}/detail/generic_primitive_get_tex_coord.h
    ${HEADER_DIR}/detail/gpu_buffer_rt.inl
    ${HEADER_DIR}/detail/macros.h
    ${HEADER_DIR}/detail/material.inl
    ${HEADER_DIR}/detail/multi_hit.h
    ${HEADER_DIR}/detail/pathtracing.inl
    ${HEADER_DIR}/detail/pixel_access.h
    ${HEADER_DIR}/detail/pixel_unpack_buffer_rt.inl
    ${HEADER_DIR}/detail/platform.h
    ${HEADER_DIR}/detail/point_light.inl
    ${HEADER_DIR}/detail/sched_common.h
    ${HEADER_DIR}/detail/semaphore.h
    ${HEADER_DIR}/detail/simple.inl
    ${HEADER_DIR}/detail/simple_buffer_rt.inl
    ${HEADER_DIR}/detail/simple_sched.h
    ${HEADER_DIR}/detail/simple_sched.inl
    ${HEADER_DIR}/detail/spectrum.inl
    ${HEADER_DIR}/detail/spot_light.inl
    ${HEADER_DIR}/detail/stack.h
    ${HEADER_DIR}/detail/surface.inl
    ${HEADER_DIR}/detail/tags.h
    ${HEADER_DIR}/detail/tiled_sched.h
    ${HEADER_DIR}/detail/tiled_sched.inl
    ${HEADER_DIR}/detail/traverse_linear.inl
    ${HEADER_DIR}/detail/whitted.inl

    # OpenGL

    ${HEADER_DIR}/gl/bvh_outline_renderer.h
    ${HEADER_DIR}/gl/compositing.h
    ${HEADER_DIR}/gl/debug_callback.h
    ${HEADER_DIR}/gl/handle.h
    ${HEADER_DIR}/gl/types.h
    ${HEADER_DIR}/gl/util.h

    # Math

    ${HEADER_DIR}/math/detail/aabb.inl
    ${HEADER_DIR}/math/detail/limits.inl
    ${HEADER_DIR}/math/detail/math.h
    ${HEADER_DIR}/math/detail/matrix4.inl
    ${HEADER_DIR}/math/detail/plane.inl
    ${HEADER_DIR}/math/detail/quaternion.inl
    ${HEADER_DIR}/math/detail/ray.inl
    ${HEADER_DIR}/math/detail/rectangle.inl
    ${HEADER_DIR}/math/detail/snorm.inl
    ${HEADER_DIR}/math/detail/sphere.inl
    ${HEADER_DIR}/math/detail/triangle.inl
    ${HEADER_DIR}/math/detail/unorm.inl
    ${HEADER_DIR}/math/detail/vector.inl
    ${HEADER_DIR}/math/detail/vector2.inl
    ${HEADER_DIR}/math/detail/vector3.inl
    ${HEADER_DIR}/math/detail/vector4.inl
    ${HEADER_DIR}/math/simd/detail/avx/int8.inl
    ${HEADER_DIR}/math/simd/detail/avx/float8.inl
    ${HEADER_DIR}/math/simd/detail/avx/mask8.inl
    ${HEADER_DIR}/math/simd/detail/avx512/int16.inl
    ${HEADER_DIR}/math/simd/detail/avx512/float16.inl
    ${HEADER_DIR}/math/simd/detail/avx512/mask16.inl
    ${HEADER_DIR}/math/simd/detail/sse/int4.inl
    ${HEADER_DIR}/math/simd/detail/sse/float4.inl
    ${HEADER_DIR}/math/simd/detail/sse/mask4.inl
    ${HEADER_DIR}/math/simd/detail/sse/matrix4.inl
    ${HEADER_DIR}/math/simd/detail/basic_float.inl
    ${HEADER_DIR}/math/simd/detail/basic_int.inl
    ${HEADER_DIR}/math/simd/detail/basic_mask.inl
    ${HEADER_DIR}/math/simd/detail/common.h
    ${HEADER_DIR}/math/simd/avx.h
    ${HEADER_DIR}/math/simd/avx512.h
    ${HEADER_DIR}/math/simd/forward.h
    ${HEADER_DIR}/math/simd/gather.h
    ${HEADER_DIR}/math/simd/intrinsics.h
    ${HEADER_DIR}/math/simd/matrix.h
    ${HEADER_DIR}/math/simd/simd.h
    ${HEADER_DIR}/math/simd/sse.h
    ${HEADER_DIR}/math/simd/trans.h
    ${HEADER_DIR}/math/simd/type_traits.h
    ${HEADER_DIR}/math/aabb.h
    ${HEADER_DIR}/math/axis.h
    ${HEADER_DIR}/math/config.h
    ${HEADER_DIR}/math/forward.h
    ${HEADER_DIR}/math/intersect.h
    ${HEADER_DIR}/math/io.h
    ${HEADER_DIR}/math/limits.h
    ${HEADER_DIR}/math/math.h
    ${HEADER_DIR}/math/matrix.h
    ${HEADER_DIR}/math/norm.h
    ${HEADER_DIR}/math/primitive.h
    ${HEADER_DIR}/math/project.h
    ${HEADER_DIR}/math/plane.h
    ${HEADER_DIR}/math/quaternion.h
    ${HEADER_DIR}/math/ray.h
    ${HEADER_DIR}/math/rectangle.h
    ${HEADER_DIR}/math/serialization.h
    ${HEADER_DIR}/math/snorm.h
    ${HEADER_DIR}/math/sphere.h
    ${HEADER_DIR}/math/triangle.h
    ${HEADER_DIR}/math/unorm.h
    ${HEADER_DIR}/math/vector.h

    # Texture access

    ${HEADER_DIR}/texture/detail/filter/common.h
    ${HEADER_DIR}/texture/detail/filter/cubic.h
    ${HEADER_DIR}/texture/detail/filter/cubic_opt.h
    ${HEADER_DIR}/texture/detail/filter/linear.h
    ${HEADER_DIR}/texture/detail/filter/nearest.h
    ${HEADER_DIR}/texture/detail/cuda_texture.h
    ${HEADER_DIR}/texture/detail/cuda_texture1d.inl
    ${HEADER_DIR}/texture/detail/cuda_texture2d.inl
    ${HEADER_DIR}/texture/detail/cuda_texture3d.inl
    ${HEADER_DIR}/texture/detail/filter.h
    ${HEADER_DIR}/texture/detail/prefilter.h
    ${HEADER_DIR}/texture/detail/sampler1d.h
    ${HEADER_DIR}/texture/detail/sampler2d.h
    ${HEADER_DIR}/texture/detail/sampler3d.h
    ${HEADER_DIR}/texture/detail/texture1d.h
    ${HEADER_DIR}/texture/detail/texture2d.h
    ${HEADER_DIR}/texture/detail/texture3d.h
    ${HEADER_DIR}/texture/detail/texture_common.h
    ${HEADER_DIR}/texture/forward.h
    ${HEADER_DIR}/texture/texture.h

    # General library headers

    ${HEADER_DIR}/aligned_vector.h
    ${HEADER_DIR}/area_light.h
    ${HEADER_DIR}/array.h
    ${HEADER_DIR}/array_ref.h
    ${HEADER_DIR}/brdf.h
    ${HEADER_DIR}/bvh.h
    ${HEADER_DIR}/camera.h
    ${HEADER_DIR}/cpu_buffer_rt.h
    ${HEADER_DIR}/exception.h
    ${HEADER_DIR}/fresnel.h
    ${HEADER_DIR}/generic_primitive.h
    ${HEADER_DIR}/generic_material.h
    ${HEADER_DIR}/get_color.h
    ${HEADER_DIR}/get_normal.h
    ${HEADER_DIR}/get_shading_normal.h
    ${HEADER_DIR}/get_surface.h
    ${HEADER_DIR}/get_tex_coord.h
    ${HEADER_DIR}/gpu_buffer_rt.h
    ${HEADER_DIR}/intersector.h
    ${HEADER_DIR}/kernels.h
    ${HEADER_DIR}/material.h
    ${HEADER_DIR}/packet_traits.h
    ${HEADER_DIR}/pixel_format.h
    ${HEADER_DIR}/pixel_traits.h
    ${HEADER_DIR}/pixel_unpack_buffer_rt.h
    ${HEADER_DIR}/point_light.h
    ${HEADER_DIR}/prim_traits.h
    ${HEADER_DIR}/random_sampler.h
    ${HEADER_DIR}/render_target.h
    ${HEADER_DIR}/result_record.h
    ${HEADER_DIR}/sampling.h
    ${HEADER_DIR}/scheduler.h
    ${HEADER_DIR}/shade_record.h
    ${HEADER_DIR}/simple_buffer_rt.h
    ${HEADER_DIR}/spectrum.h
    ${HEADER_DIR}/spot_light.h
    ${HEADER_DIR}/surface.h
    ${HEADER_DIR}/swizzle.h
    ${HEADER_DIR}/tags.h
    ${HEADER_DIR}/traverse.h
    ${HEADER_DIR}/update_if.h
    ${HEADER_DIR}/variant.h
    ${HEADER_DIR}/version.h

    #----------------------------------------------------------------------------------------------
    # Private headers
    #

    util.h

)

set(VSNRAY_SOURCES

    cuda/graphics_resource.cpp

    detail/spd/d65.cpp

    gl/bvh_outline_renderer.cpp
    gl/compositing.cpp
    gl/debug_callback.cpp
    gl/handle.cpp
    gl/util.cpp

    exception.cpp
    pixel_format.cpp
    util.cpp

)


#--------------------------------------------------------------------------------------------------
# Add library target
#

visionaray_add_library(visionaray
    ${VSNRAY_HEADERS}
    ${VSNRAY_SOURCES}
)


#--------------------------------------------------------------------------------------------------
# Install headers and library
#

install(DIRECTORY ${HEADER_DIR}
    DESTINATION include
    COMPONENT headers
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hpp"
        PATTERN "*.hxx"
        PATTERN "*.inl"
        PATTERN "*.inc"
)

install(TARGETS visionaray
     RUNTIME DESTINATION bin COMPONENT libraries
     LIBRARY DESTINATION lib COMPONENT libraries
     ARCHIVE DESTINATION lib COMPONENT libraries
)
