// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/cuda/managed_vector.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/sphere.h>
#include <visionaray/bvh.h>
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/result_record.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>

#include <common/timer.h>

using namespace visionaray;


namespace visionaray
{
namespace cuda
{

//-------------------------------------------------------------------------------------------------
// Typedef so we can just write cuda::managed_bvh
//

template <typename P>
using managed_bvh = index_bvh_t<managed_vector<P>, managed_vector<bvh_node>, managed_vector<unsigned>>;

} // cuda
} // visionaray

template <typename Cont>
static void create_random_spheres(Cont& spheres, aabb bbox, float min_radius, float max_radius)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist_x(bbox.min.x, bbox.max.x);
    std::uniform_real_distribution<float> dist_y(bbox.min.y, bbox.max.y);
    std::uniform_real_distribution<float> dist_z(bbox.min.z, bbox.max.z);
    std::uniform_real_distribution<float> dist_r(min_radius, max_radius);


    // Spheres
    for (size_t i = 0; i < spheres.size(); ++i)
    {
        spheres[i] = typename Cont::value_type(vec3(dist_x(mt), dist_y(mt), dist_z(mt)), dist_r(mt));
        spheres[i].prim_id = static_cast<int>(i);
        spheres[i].geom_id = static_cast<int>(i);
    }
}


//-------------------------------------------------------------------------------------------------
// Most simple Visionaray ray tracing kernel
//

template <typename It>
struct raytracing_kernel
{
    __host__ __device__
    result_record<float> operator()(ray r)
    {
        auto hr = closest_hit(r, first, last);

        result_record<float> result;
        result.hit = hr.hit;
        result.color = hr.hit ? vec4(1.0f) : vec4(0.0f);
        result.isect_pos = r.ori + r.dir * hr.t;
        return result;
    }

    It first;
    It last;
};


//-------------------------------------------------------------------------------------------------
// Main function
//

int main(int argc, char** argv)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    size_t num_spheres = 300000;
    aabb bbox({ -1.0f, -1.0f, -1.0f}, { 1.0f, 1.0f, 1.0f });
    float min_radius = 0.001f;
    float max_radius = 0.002f;
    int width = 512;
    int height = 512;

    bool unified_memory_mode = true;

    if (unified_memory_mode)
    {
        // Don't measure runtime API initialization overhead
        cudaDeviceSynchronize();

        std::cout << "\n*** CUDA unified memory example ***\n\n";
        std::cout << "Using memory mode: CUDA unified memory\n\n";


        // Create data in unified memory ------------------

        cuda::timer t;
        cuda::managed_vector<basic_sphere<float>> spheres(num_spheres);

        std::cout << "Creating " << num_spheres << " random spheres...\n";
        create_random_spheres(spheres, bbox, min_radius, max_radius);
        cudaDeviceSynchronize();
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Create BVH -------------------------------------

        std::cout << "Creating BVH...\n";
        t.reset();
        auto bvh = build<cuda::managed_bvh<basic_sphere<float>>>(spheres.data(), spheres.size(), true /* spatial splits */);
        cudaDeviceSynchronize();
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Prepare for ray tracing ------------------------

        using bvh_ref_t = typename cuda::managed_bvh<basic_sphere<float>>::bvh_ref;
        cuda::managed_vector<bvh_ref_t> bvh_refs(1);
        bvh_refs[0] = bvh.ref();

        pinhole_camera cam;
        cam.set_viewport(0, 0, width, height);
        cam.perspective(45.0f * constants::degrees_to_radians<float>(), 1.0f, 0.001f, 1000.0f);
        cam.view_all(bbox);

        gpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED> rendertarget;
        rendertarget.resize(width, height);

        auto sparams = make_sched_params(cam, rendertarget);

        raytracing_kernel<bvh_ref_t const*> kern = { bvh_refs.data(), bvh_refs.data() + 1 };

        cuda_sched<ray> sched;


        // Ray tracing on the GPU -------------------------

        std::cout << "Calculating primary visibility with " << width << " x " << height << " rays...\n";
        t.reset();
        sched.frame(kern, sparams);
        cudaDeviceSynchronize();
        std::cout << "Time eplased: " << t.elapsed() << "s\n\n";
    }
}
