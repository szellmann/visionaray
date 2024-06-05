// This file is distributed under the MIT license.
// See the LICENSE file for details.

//-------------------------------------------------------------------------------------------------
// This file is based on Kevin Beason's smallpt global illumination renderer.
// Original license (MIT) follows.
//

/*
LICENSE

Copyright (c) 2006-2008 Kevin Beason (kevin.beason@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <exception>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <thread>

#include <GL/glew.h>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/math/sphere.h>

#include <visionaray/area_light.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/scheduler.h>

#include <common/input/key_event.h>
#include <common/cpu_buffer_rt.h>
#include <common/make_materials.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <common/pixel_unpack_buffer_rt.h>
#endif

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Switch between single and double precision implementation
//

#define USE_DOUBLE_PRECISION 1


//-------------------------------------------------------------------------------------------------
// Switch between path tracing with MIS and naive path tracing
//

#define USE_MULTIPLE_IMPORTANCE_SAMPLING 0


//-------------------------------------------------------------------------------------------------
// Visionaray camera generating primary rays similar to how smallpt does
//

struct smallpt_camera
{
    VSNRAY_FUNC void begin_frame() {}
    VSNRAY_FUNC void end_frame() {}

    template <typename R, typename T>
    VSNRAY_FUNC
    inline R primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const
    {
        using V = vector<3, T>;

        V eye(50.0, 52.0, 295.6);
        V dir = normalize(V(0.0, -0.042612, -1.0));
        V cx(width * 0.5135 / height, 0.0, 0.0);
        V cy = normalize(cross(cx, dir)) * T(0.5135);

        V d = cx * ((x - width / T(2.0)) / width) + cy * ((y - height / T(2.0)) / height) + dir;
        return R(eye + d * T(140.0), normalize(d));
    }
};


//-------------------------------------------------------------------------------------------------
// Renderer
//

struct renderer : viewer_type
{
#if USE_DOUBLE_PRECISION
    using S   = double;
    using Vec = vec3d;
#else
    using S   = float;
    using Vec = vec3f;
#endif

    using host_ray_type   = basic_ray<S>;
    using device_ray_type = basic_ray<S>;
    using material_type   = generic_material<emissive<S>, glass<S>, matte<S>, mirror<S>>;
    using area_light_type = area_light<S, basic_sphere<S>>;

    enum device_type
    {
        CPU = 0,
        GPU
    };

    struct empty_light_type {};

    renderer()
        : viewer_type(512, 512, "Visionaray Smallpt Example")
        , host_sched(std::thread::hardware_concurrency())
#ifdef __CUDACC__
        , device_sched(8, 8)
#endif
    {
#ifdef __CUDACC__
        using namespace support;

        add_cmdline_option( cl::makeOption<device_type&>({
                { "cpu", CPU, "Rendering on the CPU" },
                { "gpu", GPU, "Rendering on the GPU" },
            },
            "device",
            cl::Desc("Rendering device"),
            cl::ArgRequired,
            cl::init(this->dev_type)
            ) );
#endif
    }

    void init_scene()
    {
#if USE_DOUBLE_PRECISION
        S base_size = 1e5;
#else
        S base_size = 1e4;
#endif

        // Left
        spheres.push_back(make_sphere(Vec(base_size + 1.0, 40.8, 81.6), base_size));
        materials.push_back(make_matte(Vec(0.75, 0.25, 0.25)));

        // Right
        spheres.push_back(make_sphere(Vec(-base_size + 99.0, 40.8, 81.6), base_size));
        materials.push_back(make_matte(Vec(0.25, 0.25, 0.75)));

        // Back
        spheres.push_back(make_sphere(Vec(50.0, 40.8, base_size), base_size));
        materials.push_back(make_matte(Vec(0.75, 0.75, 0.75)));

        // Front
        spheres.push_back(make_sphere(Vec(50.0, 40.8, -base_size + 170), base_size));
        materials.push_back(make_matte(Vec(0.0)));

        // Bottom
        spheres.push_back(make_sphere(Vec(50.0, base_size, 81.6), base_size));
        materials.push_back(make_matte(Vec(0.75, 0.75, 0.75)));

        // Top
        spheres.push_back(make_sphere(Vec(50.0, -base_size + 81.6, 81.6), base_size));
        materials.push_back(make_matte(Vec(0.75, 0.75, 0.75)));

        // Mirror
        spheres.push_back(make_sphere(Vec(27.0, 16.5, 47.0), 16.5));
        materials.push_back(make_mirror(Vec(0.999)));

        // Glass
        spheres.push_back(make_sphere(Vec(73.0, 16.5, 78.0), 16.5));
        materials.push_back(make_glass(Vec(0.999)));

        // Light
        spheres.push_back(make_sphere(Vec(50.0, 681.6 - 0.27, 81.6), 600.0));
        materials.push_back(make_emissive(Vec(12.0, 12.0, 12.0)));

#if USE_MULTIPLE_IMPORTANCE_SAMPLING
        area_light_type al(spheres.back());
        al.set_cl(Vec(12.0, 12.0, 12.0));
        al.set_kl(1.0);
        lights.push_back(al);
#endif

#ifdef __CUDACC__
        // Copy to GPU
        device_spheres = spheres;
        device_materials = materials;
        device_lights = lights;
#endif
    }

    smallpt_camera                                      cam;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> host_rt;
    tiled_sched<host_ray_type>                          host_sched;

    aligned_vector<basic_sphere<S>>                     spheres;
    aligned_vector<material_type>                       materials;
    aligned_vector<area_light_type>                     lights;

#ifdef __CUDACC__
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>  device_rt;
    cuda_sched<device_ray_type>                         device_sched;

    thrust::device_vector<basic_sphere<S>>              device_spheres;
    thrust::device_vector<material_type>                device_materials;
    thrust::device_vector<area_light_type>              device_lights;
#endif

    unsigned                                            frame_num = 0;
    double                                              avg_frame_time = 0.0;

    device_type                                         dev_type = CPU;

protected:

    void on_display();
    void on_key_press(key_event const& event);
    void on_resize(int w, int h);

private:

    basic_sphere<S> make_sphere(Vec center, S radius)
    {
        static int sphere_id = 0;
        basic_sphere<S> sphere(center, radius);
        sphere.prim_id = sphere_id;
        sphere.geom_id = sphere_id;
        ++sphere_id;
        return sphere;
    }

    emissive<S> make_emissive(Vec ce)
    {
        emissive<S> mat;
        mat.ce() = from_rgb(ce);
        mat.ls() = S(1.0);
        return mat;
    }

    glass<S> make_glass(Vec ct)
    {
        glass<S> mat;
        mat.ct() = from_rgb(ct);
        mat.kt() = S(1.0);
        mat.cr() = from_rgb(ct);
        mat.kr() = S(1.0);
        mat.ior() = spectrum<S>(1.5);
        return mat;
    }

    matte<S> make_matte(Vec cd)
    {
        matte<S> mat;
        mat.cd() = from_rgb(cd);
        mat.kd() = S(1.0);
        return mat;
    }

    mirror<S> make_mirror(Vec cr)
    {
        mirror<S> mat;
        mat.cr() = from_rgb(cr);
        mat.kr() = S(0.9);
        mat.ior() = spectrum<S>(0.0);
        mat.absorption() = spectrum<S>(0.0);
        return mat;
    }
};


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // Enable gamma correction with OpenGL.
    glEnable(GL_FRAMEBUFFER_SRGB);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


#if !USE_MULTIPLE_IMPORTANCE_SAMPLING
    empty_light_type* ignore = 0;
#endif

#if USE_DOUBLE_PRECISION
    float epsilon = 1e-4f;
#else
    float epsilon = 1.5e-2f;
#endif

    double elapsed = 0.0;

#if USE_DOUBLE_PRECISION
    pixel_sampler::basic_jittered_blend_type<double> blend_params;
    double alpha = 1.0 / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0 - alpha;
#else
    pixel_sampler::basic_jittered_blend_type<float> blend_params;
    float alpha = 1.0f / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0f - alpha;
#endif

    if (dev_type == GPU)
    {
#ifdef __CUDACC__
        auto sparams = make_sched_params(
                blend_params,
                cam,
                device_rt
                ); 

        auto kparams = make_kernel_params(
                thrust::raw_pointer_cast(device_spheres.data()),
                thrust::raw_pointer_cast(device_spheres.data()) + device_spheres.size(),
                thrust::raw_pointer_cast(device_materials.data()),
#if USE_MULTIPLE_IMPORTANCE_SAMPLING
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
#else
                ignore,
                ignore,
#endif
                10,
                epsilon,
                vec4(background_color(), 1.0f),
                vec4(0.0f)
                );

        visionaray::cuda::timer t;

        pathtracing::kernel<decltype(kparams)> kernel;
        kernel.params = kparams;
        device_sched.frame(kernel, sparams);

        elapsed = t.elapsed();
        avg_frame_time += elapsed;

        device_rt.display_color_buffer();
#endif
    }
    else if (dev_type == CPU)
    {
#ifndef __CUDA_ARCH__
        auto sparams = make_sched_params(
                blend_params,
                cam,
                host_rt
                ); 

        auto kparams = make_kernel_params(
                spheres.data(),
                spheres.data() + spheres.size(),
                materials.data(),
#if USE_MULTIPLE_IMPORTANCE_SAMPLING
                lights.data(),
                lights.data() + lights.size(),
#else
                ignore,
                ignore,
#endif
                10,
                epsilon,
                vec4(background_color(), 1.0f),
                vec4(0.0f)
                );

        timer t;

        pathtracing::kernel<decltype(kparams)> kernel;
        kernel.params = kparams;
        host_sched.frame(kernel, sparams);

        elapsed = t.elapsed();
        avg_frame_time += elapsed;

        host_rt.display_color_buffer();
#endif
    }

    std::cout << std::setprecision(3);
    std::cout << std::fixed;

    std::cout << "Num frames: " << frame_num
              << ", last frame time: " << elapsed
              << ", avg. frame time: " << avg_frame_time / frame_num << '\r';
    std::cout << std::flush;
}


//-------------------------------------------------------------------------------------------------
// Key press event
//

void renderer::on_key_press(key_event const& event)
{
    switch (event.key())
    {
    case 'm':
#ifdef __CUDACC__
        if (dev_type == renderer::CPU)
        {
            dev_type = renderer::GPU;
        }
        else
        {
            dev_type = renderer::CPU;
        }

        avg_frame_time = 0.0;
        frame_num = 0;
        host_rt.clear_color_buffer();
        device_rt.clear_color_buffer();
#endif
        break;
    default:
        break;
    }

    viewer_type::on_key_press(event);
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    frame_num = 0;

    host_rt.resize(w, h);
    host_rt.clear_color_buffer();
#ifdef __CUDACC__
    device_rt.resize(w, h);
    device_rt.clear_color_buffer();
#endif

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
#ifdef __CUDACC__
    int device = 0;
    cudaDeviceProp prop;
    if (cudaChooseDevice(&device, &prop) != cudaSuccess)
    {   
        std::cerr << "Cannot choose CUDA device " << device << '\n';
        return EXIT_FAILURE;
    }   
#endif

    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    rend.init_scene();

    rend.event_loop();
}
