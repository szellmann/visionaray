// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/kernels.h> // make_kernel_params()

#include "render.h"

namespace visionaray
{

void render_plastic_cpp(
        index_bvh<basic_triangle<3, float>> const& bvh,
        aligned_vector<vec3> const&                geometric_normals,
        aligned_vector<vec3> const&                shading_normals,
        aligned_vector<vec2> const&                tex_coords,
        aligned_vector<plastic_t> const&           materials,
        aligned_vector<texture_t> const&           textures,
        aligned_vector<point_light<float>> const&  lights,
        camera_t const&                            cam,
        host_device_rt&                            rt,
        host_sched_t<ray_type_cpu>&                sched,
        unsigned&                                  frame_num,
        render_state                               state
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
            state.bounces,
            state.epsilon,
            state.bgcolor,
            state.ambient
            );

    call_kernel(state.algo, sched, kparams, frame_num, state.num_samples, cam, rt);
}

} // visionaray
