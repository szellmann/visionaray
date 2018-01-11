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

#include <visionaray/math/sphere.h>

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/scheduler.h>

#include <common/make_materials.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Visionaray camera generating primary rays similar to how smallpt does
//

struct smallpt_camera
{
    void begin_frame() {}
    void end_frame() {}

    template <typename R, typename T>
    inline R primary_ray(R /* */, T const& x, T const& y, T const& width, T const& height) const
    {
        using V = vector<3, T>;

        V eye(50.0, 52.0, 295.6);
        V dir = normalize(V(0.0, -0.042612, -1.0));
        V cx(width * 0.5135 / height, 0.0, 0.0);
        V cy = normalize(cross(cx, dir)) * T(0.5135);

        V d = cx * ((x - width / 2.0) / width) + cy * ((y - height / 2.0) / height) + dir;
        return R(eye + d * 140.0, normalize(d));
    }
};


//-------------------------------------------------------------------------------------------------
// Renderer
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<double>;
    using material_type = generic_material<emissive<double>, glass<double>, matte<double>, mirror<double>>;

    renderer()
        : viewer_type(512, 512, "Visionaray Smallpt Example")
        , host_sched(std::thread::hardware_concurrency())
    {
    }

    void init_scene()
    {
        // Left
        spheres.push_back(make_sphere(vec3d(1e5 + 1.0, 40.8, 81.6), 1e5));
        materials.push_back(make_matte(vec3d(0.75, 0.25, 0.25)));

        // Right
        spheres.push_back(make_sphere(vec3d(-1e5 + 99.0, 40.8, 81.6), 1e5));
        materials.push_back(make_matte(vec3d(0.25, 0.25, 0.75)));

        // Back
        spheres.push_back(make_sphere(vec3d(50.0, 40.8, 1e5), 1e5));
        materials.push_back(make_matte(vec3d(0.75, 0.75, 0.75)));

        // Front
        spheres.push_back(make_sphere(vec3d(50.0, 40.8, -1e5 + 170), 1e5));
        materials.push_back(make_matte(vec3d(0.0)));

        // Bottom
        spheres.push_back(make_sphere(vec3d(50.0, 1e5, 81.6), 1e5));
        materials.push_back(make_matte(vec3d(0.75, 0.75, 0.75)));

        // Top
        spheres.push_back(make_sphere(vec3d(50.0, -1e5 + 81.6, 81.6), 1e5));
        materials.push_back(make_matte(vec3d(0.75, 0.75, 0.75)));

        // Mirror
        spheres.push_back(make_sphere(vec3d(27.0, 16.5, 47.0), 16.5));
        materials.push_back(make_mirror(vec3d(0.999)));

        // Glass
        spheres.push_back(make_sphere(vec3d(73.0, 16.5, 78.0), 16.5));
        materials.push_back(make_glass(vec3d(0.999)));

        // Light
        spheres.push_back(make_sphere(vec3d(50.0, 681.6 - 0.27, 81.6), 600.0));
        materials.push_back(make_emissive(vec3d(12.0, 12.0, 12.0)));
    }

    smallpt_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    unsigned                                    frame_num = 0;

    aligned_vector<basic_sphere<double>>        spheres;
    aligned_vector<material_type>               materials;

protected:

    void on_display();
    void on_resize(int w, int h);

private:

    basic_sphere<double> make_sphere(vec3d center, double radius)
    {
        static int sphere_id = 0;
        basic_sphere<double> sphere(center, radius);
        sphere.prim_id = sphere_id;
        sphere.geom_id = sphere_id;
        ++sphere_id;
        return sphere;
    }

    emissive<double> make_emissive(vec3d ce)
    {
        emissive<double> mat;
        mat.ce() = from_rgb(ce);
        mat.ls() = 1.0;
        return mat;
    }

    glass<double> make_glass(vec3d ct)
    {
        glass<double> mat;
        mat.ct() = from_rgb(ct);
        mat.kt() = 1.0;
        mat.cr() = from_rgb(ct);
        mat.kr() = 1.0;
        mat.ior() = spectrum<double>(1.5);
        return mat;
    }

    matte<double> make_matte(vec3d cd)
    {
        matte<double> mat;
        mat.cd() = from_rgb(cd);
        mat.kd() = 1.0;
        return mat;
    }

    mirror<double> make_mirror(vec3d cr)
    {
        mirror<double> mat;
        mat.cr() = from_rgb(cr);
        mat.kr() = 0.9;
        mat.ior() = spectrum<double>(0.0);
        mat.absorption() = spectrum<double>(0.0);
        return mat;
    }
};


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    auto sparams = make_sched_params(
            pixel_sampler::jittered_blend_type{},
            cam,
            host_rt
            ); 

    // make_kernel_params needs (!) lights
    // TODO: fix this in visionaray API!
    struct no_lights {};
    no_lights* ignore = 0;

    auto kparams = make_kernel_params(
            spheres.data(),
            spheres.data() + spheres.size(),
            materials.data(),
            ignore,
            ignore,
            10,
            1e-4f, // epsilon
            vec4(background_color(), 1.0f),
            vec4(0.0f)
            );

    pathtracing::kernel<decltype(kparams)> kernel;
    kernel.params = kparams;
    host_sched.frame(kernel, sparams, ++frame_num);


    // Enable gamma correction with OpenGL.
    glEnable(GL_FRAMEBUFFER_SRGB);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();

    std::cout << "Num samples: " << frame_num << '\r';
    std::cout << std::flush;
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    frame_num = 0;

    host_rt.clear_color_buffer();
    host_rt.resize(w, h);

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
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