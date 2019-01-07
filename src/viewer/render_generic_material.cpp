// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/kernels.h> // make_kernel_params()

#include "render.h"

namespace visionaray
{

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
        )
{
    using bvh_ref = index_bvh<basic_triangle<3, float>>::bvh_ref;

    aligned_vector<bvh_ref> primitives;

    primitives.push_back(bvh.ref());

    auto kparams = make_kernel_params(
            normals_per_vertex_binding{},
            primitives.data(),
            primitives.data() + primitives.size(),
            geometric_normals.data(),
            shading_normals.data(),
            tex_coords.data(),
            materials.data(),
            textures.data(),
            lights.data(),
            lights.data() + lights.size(),
            bounces,
            epsilon,
            bgcolor,
            ambient
            );

    call_kernel( algo, sched, kparams, frame_num, ssaa_samples, cam, rt );
}

} // visionaray
