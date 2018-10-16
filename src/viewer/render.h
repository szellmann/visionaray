// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_RENDER_H
#define VSNRAY_VIEWER_RENDER_H 1

#include <common/config.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/ray.h>
#include <visionaray/math/sphere.h>
#include <visionaray/math/triangle.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/area_light.h>
#include <visionaray/bvh.h>
#include <visionaray/generic_light.h>
#include <visionaray/generic_material.h>
#include <visionaray/material.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>
#include <visionaray/spot_light.h>
#include <visionaray/thin_lens_camera.h>
#include <visionaray/variant.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#if VSNRAY_COMMON_HAVE_PTEX
#include <common/ptex.h>
#endif

#include "call_kernel.h" // algorithm
#include "host_device_rt.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Explicit template instantiation of render calls for faster parallel builds
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Helper types
//

//using scalar_type_cpu           = float;
using scalar_type_cpu           = simd::float4;
//using scalar_type_cpu           = simd::float8;
//using scalar_type_cpu           = simd::float16;
using scalar_type_gpu           = float;
using ray_type_cpu              = basic_ray<scalar_type_cpu>;
using ray_type_gpu              = basic_ray<scalar_type_gpu>;

using camera_t = variant<pinhole_camera, thin_lens_camera>;
using plastic_t = plastic<float>;
using generic_light_t = generic_light<
        point_light<float>,
        spot_light<float>,
        area_light<float, basic_triangle<3, float>>//,
//      area_light<float, basic_sphere<float>>
        >;
using generic_material_t = generic_material<
        emissive<float>,
        glass<float>,
        matte<float>,
        mirror<float>,
        plastic<float>
        >;
using texture_t = texture_ref<vector<4, unorm<8>>, 2>;
#ifdef __CUDACC__
using cuda_texture_t = cuda_texture_ref<vector<4, unorm<8>>, 2>;
#endif

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
template <typename R>
using host_sched_t = tbb_sched<R>;
#else
template <typename R>
using host_sched_t = tiled_sched<R>;
#endif


//-------------------------------------------------------------------------------------------------
// Render from lists, only material is plastic
//

void render_plastic_cpp(
        index_bvh<basic_triangle<3, float>> const& bvh,
        aligned_vector<vec3> const&                geometric_normals,
        aligned_vector<vec3> const&                shading_normals,
        aligned_vector<vec2> const&                tex_coords,
        aligned_vector<plastic_t> const&           materials,
        aligned_vector<texture_t> const&           textures,
        aligned_vector<point_light<float>> const&  lights,
        unsigned                                   bounces,
        float                                      epsilon,
        vec4                                       bgcolor,
        vec4                                       ambient,
        host_device_rt&                            rt,
        host_sched_t<ray_type_cpu>&                sched,
        camera_t const&                            cam,
        unsigned&                                  frame_num,
        algorithm                                  algo,
        unsigned                                   ssaa_samples
        );

#ifdef __CUDACC__
void render_plastic_cu(
        cuda_index_bvh<basic_triangle<3, float>>&         bvh,
        thrust::device_vector<vec3> const&                geometric_normals,
        thrust::device_vector<vec3> const&                shading_normals,
        thrust::device_vector<vec2> const&                tex_coords,
        thrust::device_vector<plastic_t> const&           materials,
        thrust::device_vector<cuda_texture_t> const&      textures,
        aligned_vector<point_light<float>> const&         host_lights,
        unsigned                                          bounces,
        float                                             epsilon,
        vec4                                              bgcolor,
        vec4                                              ambient,
        host_device_rt&                                   rt,
        cuda_sched<ray_type_gpu>&                         sched,
        camera_t const&                                   cam,
        unsigned&                                         frame_num,
        algorithm                                         algo,
        unsigned                                          ssaa_samples
        );
#endif


//-------------------------------------------------------------------------------------------------
// Render from lists, material is generic
//

void render_generic_material_cpp(
        index_bvh<basic_triangle<3, float>> const&                         bvh,
        aligned_vector<vec3> const&                                        geometric_normals,
        aligned_vector<vec3> const&                                        shading_normals,
        aligned_vector<vec2> const&                                        tex_coords,
        aligned_vector<generic_material_t> const&                          materials,
        aligned_vector<texture_t> const&                                   textures,
        aligned_vector<area_light<float, basic_triangle<3, float>>> const& lights,
        unsigned                                                           bounces,
        float                                                              epsilon,
        vec4                                                               bgcolor,
        vec4                                                               ambient,
        host_device_rt&                                                    rt,
        host_sched_t<ray_type_cpu>&                                        sched,
        camera_t const&                                                    cam,
        unsigned&                                                          frame_num,
        algorithm                                                          algo,
        unsigned                                                           ssaa_samples
        );

#ifdef __CUDACC__
void render_generic_material_cu(
        cuda_index_bvh<basic_triangle<3, float>>&                          bvh,
        thrust::device_vector<vec3> const&                                 geometric_normals,
        thrust::device_vector<vec3> const&                                 shading_normals,
        thrust::device_vector<vec2> const&                                 tex_coords,
        thrust::device_vector<generic_material_t> const&                   materials,
        thrust::device_vector<cuda_texture_t> const&                       textures,
        aligned_vector<area_light<float, basic_triangle<3, float>>> const& lights,
        unsigned                                                           bounces,
        float                                                              epsilon,
        vec4                                                               bgcolor,
        vec4                                                               ambient,
        host_device_rt&                                                    rt,
        cuda_sched<ray_type_gpu>&                                          sched,
        camera_t const&                                                    cam,
        unsigned&                                                          frame_num,
        algorithm                                                          algo,
        unsigned                                                           ssaa_samples
        );
#endif


//-------------------------------------------------------------------------------------------------
// Render mesh instances (everything else is generic!)
//

void render_instances_cpp(
        index_bvh<index_bvh<basic_triangle<3, float>>::bvh_inst>& bvh,
        aligned_vector<vec3> const&                               geometric_normals,
        aligned_vector<vec3> const&                               shading_normals,
        aligned_vector<vec2> const&                               tex_coords,
        aligned_vector<generic_material_t> const&                 materials,
        aligned_vector<texture_t> const&                          textures,
        aligned_vector<generic_light_t> const&                    lights,
        unsigned                                                  bounces,
        float                                                     epsilon,
        vec4                                                      bgcolor,
        vec4                                                      ambient,
        host_device_rt&                                           rt,
        host_sched_t<ray_type_cpu>&                               sched,
        camera_t const&                                           cam,
        unsigned&                                                 frame_num,
        algorithm                                                 algo,
        unsigned                                                  ssaa_samples
        );

#if VSNRAY_COMMON_HAVE_PTEX
// With ptex textures
void render_instances_ptex_cpp(
        index_bvh<index_bvh<basic_triangle<3, float>>::bvh_inst>& bvh,
        aligned_vector<vec3> const&                               geometric_normals,
        aligned_vector<vec3> const&                               shading_normals,
        aligned_vector<ptex::face_id_t> const&                    face_ids,
        aligned_vector<generic_material_t> const&                 materials,
        aligned_vector<ptex::texture> const&                      textures,
        aligned_vector<generic_light_t> const&                    lights,
        unsigned                                                  bounces,
        float                                                     epsilon,
        vec4                                                      bgcolor,
        vec4                                                      ambient,
        host_device_rt&                                           rt,
        host_sched_t<ray_type_cpu>&                               sched,
        camera_t const&                                           cam,
        unsigned&                                                 frame_num,
        algorithm                                                 algo,
        unsigned                                                  ssaa_samples
        );
#endif

} // visionaray

#endif // VSNRAY_VIEWER_RENDER_H
