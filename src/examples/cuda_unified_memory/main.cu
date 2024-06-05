// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/cuda/managed_vector.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/sphere.h>
#include <visionaray/bvh.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/result_record.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>

#include <common/gpu_buffer_rt.h>
#include <common/timer.h>

using namespace support;
using namespace visionaray;

using cmdline_options = std::vector<std::shared_ptr<cl::OptionBase>>;


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
        result.depth = hr.t;
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
    // Application state ----------------------------------

    size_t num_spheres = 300000;
    aabb bbox({ -1.0f, -1.0f, -1.0f}, { 1.0f, 1.0f, 1.0f });
    float min_radius = 0.001f;
    float max_radius = 0.002f;
    int width = 512;
    int height = 512;

    bool explicit_copy_mode = false;


    // Setup ----------------------------------------------

    std::cout << std::fixed;
    std::cout << std::setprecision(4);


    // Read command line options
    cl::CmdLine cmd;
    cmdline_options options;

    options.emplace_back( cl::makeOption<size_t&>(
            cl::Parser<>(),
            "num_spheres",
            cl::Desc("Number of random spheres to traverse"),
            cl::ArgRequired,
            cl::init(num_spheres)
            ) );

    options.emplace_back( cl::makeOption<bool&>(
            cl::Parser<>(),
            "explicit_copy",
            cl::Desc("Use explicit memory transfers instead of unified memory"),
            cl::ArgDisallowed,
            cl::init(explicit_copy_mode)
            ) );

    options.emplace_back( cl::makeOption<int&>(
            cl::Parser<>(),
            "width",
            cl::Desc("Image width"),
            cl::ArgRequired,
            cl::init(width)
            ) );

    options.emplace_back( cl::makeOption<int&>(
            cl::Parser<>(),
            "height",
            cl::Desc("Image height"),
            cl::ArgRequired,
            cl::init(height)
            ) );

    for (auto& opt : options)
    {
        cmd.add(*opt);
    }

    auto args = std::vector<std::string>(argv + 1, argv + argc);
    cl::expandWildcards(args);
    cl::expandResponseFiles(args, cl::TokenizeUnix());

    try
    {
        cmd.parse(args);
    }
    catch (...)
    {
        std::cout << cmd.help(argv[0]) << '\n';
        exit(EXIT_FAILURE);
    }


    // Don't measure runtime API initialization overhead
    cudaDeviceSynchronize();

    std::cout << "\n*** CUDA unified memory example ***\n\n";

    if (!explicit_copy_mode)
    {
        std::cout << "Using memory mode: CUDA unified memory\n\n";


        // Create data in unified memory ------------------

        visionaray::cuda::timer t;
        visionaray::cuda::managed_vector<basic_sphere<float>> spheres(num_spheres);

        std::cout << "Creating " << num_spheres << " random spheres...\n";
        create_random_spheres(spheres, bbox, min_radius, max_radius);
        cudaDeviceSynchronize();
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Create BVH -------------------------------------

        std::cout << "Creating BVH...\n";
        t.reset();
        binned_sah_builder builder;
        auto bvh = builder.build(visionaray::cuda::managed_bvh<basic_sphere<float>>{}, spheres.data(), spheres.size(), true /* spatial splits */);
        cudaDeviceSynchronize();
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Prepare for ray tracing ------------------------

        using bvh_ref_t = typename visionaray::cuda::managed_bvh<basic_sphere<float>>::bvh_ref;
        visionaray::cuda::managed_vector<bvh_ref_t> bvh_refs(1);
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
    else
    {
        std::cout << "Using memory mode: explicit memory transfers\n\n";


        // Create data in host memory ---------------------

        timer t;
        thrust::host_vector<basic_sphere<float>> spheres(num_spheres);

        std::cout << "Creating " << num_spheres << " random spheres...\n";
        create_random_spheres(spheres, bbox, min_radius, max_radius);
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Create BVH -------------------------------------

        std::cout << "Creating BVH...\n";
        t.reset();
        binned_sah_builder builder;
        auto h_bvh = builder.build(index_bvh<basic_sphere<float>>{}, spheres.data(), spheres.size(), true /* spatial splits */);
        std::cout << "Time elapsed: " << t.elapsed() << "s\n\n";


        // Upload data to GPU -----------------------------

        cuda_index_bvh<basic_sphere<float>> d_bvh(h_bvh);


        // Prepare for ray tracing ------------------------

        using bvh_ref_t = typename cuda_index_bvh<basic_sphere<float>>::bvh_ref;
        thrust::device_vector<bvh_ref_t> bvh_refs;
        bvh_refs.push_back(d_bvh.ref());

        pinhole_camera cam;
        cam.set_viewport(0, 0, width, height);
        cam.perspective(45.0f * constants::degrees_to_radians<float>(), 1.0f, 0.001f, 1000.0f);
        cam.view_all(bbox);

        gpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED> rendertarget;
        rendertarget.resize(width, height);

        auto sparams = make_sched_params(cam, rendertarget);

        raytracing_kernel<bvh_ref_t const*> kern = {
                thrust::raw_pointer_cast(bvh_refs.data()),
                thrust::raw_pointer_cast(bvh_refs.data()) + 1
                };

        cuda_sched<ray> sched;


        // Ray tracing on the GPU -------------------------

        std::cout << "Calculating primary visibility with " << width << " x " << height << " rays...\n";
        visionaray::cuda::timer ct;
        sched.frame(kern, sparams);
        std::cout << "Time eplased: " << ct.elapsed() << "s\n\n";
    }
}
