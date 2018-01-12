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
#include <iostream>
#include <ostream>
#include <thread>

#include <GL/glew.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/math/sphere.h>

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/scheduler.h>

#ifdef __CUDACC__
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/make_materials.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <common/cuda.h>
#endif

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Switch between single and double precision implementation
//

#define USE_DOUBLE_PRECISION 1


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

#ifdef __CUDACC__
        // Copy to GPU
        device_spheres = spheres;
        device_materials = materials;
#endif
    }

    smallpt_camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>           host_rt;
    tiled_sched<host_ray_type>                          host_sched;

    aligned_vector<basic_sphere<S>>                     spheres;
    aligned_vector<material_type>                       materials;

#ifdef __CUDACC__
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>  device_rt;
    cuda_sched<device_ray_type>                         device_sched;

    thrust::device_vector<basic_sphere<S>>              device_spheres;
    thrust::device_vector<material_type>                device_materials;
#endif

    unsigned                                            frame_num = 0;

    device_type                                         dev_type = CPU;

protected:

    void on_display();
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


    // make_kernel_params needs (!) lights
    // TODO: fix this in visionaray API!
    empty_light_type* ignore = 0;

#if USE_DOUBLE_PRECISION
    float epsilon = 1e-4f;
#else
    float epsilon = 1.5e-2f;
#endif

    if (dev_type == GPU)
    {
#ifdef __CUDACC__
        auto sparams = make_sched_params(
                pixel_sampler::jittered_blend_type{},
                cam,
                device_rt
                ); 

        auto kparams = make_kernel_params(
                thrust::raw_pointer_cast(device_spheres.data()),
                thrust::raw_pointer_cast(device_spheres.data()) + device_spheres.size(),
                thrust::raw_pointer_cast(device_materials.data()),
                ignore,
                ignore,
                10,
                epsilon,
                vec4(background_color(), 1.0f),
                vec4(0.0f)
                );

        pathtracing::kernel<decltype(kparams)> kernel;
        kernel.params = kparams;
        device_sched.frame(kernel, sparams, ++frame_num);

        device_rt.display_color_buffer();
#endif
    }
    else if (dev_type == CPU)
    {
#ifndef __CUDA_ARCH__
        auto sparams = make_sched_params(
                pixel_sampler::jittered_blend_type{},
                cam,
                host_rt
                ); 

        auto kparams = make_kernel_params(
                spheres.data(),
                spheres.data() + spheres.size(),
                materials.data(),
                ignore,
                ignore,
                10,
                epsilon,
                vec4(background_color(), 1.0f),
                vec4(0.0f)
                );

        pathtracing::kernel<decltype(kparams)> kernel;
        kernel.params = kparams;
        host_sched.frame(kernel, sparams, ++frame_num);

        host_rt.display_color_buffer();
#endif
    }

    std::cout << "Num samples: " << frame_num << '\r';
    std::cout << std::flush;
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
    if (cuda::init_gl_interop() != cudaSuccess)
    {   
        std::cerr << "Cannot initialize CUDA OpenGL interop\n";
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
