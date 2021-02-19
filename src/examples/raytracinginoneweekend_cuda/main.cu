// This file is distributed under the MIT license.
// See the LICENSE file for details.

//-------------------------------------------------------------------------------------------------
// This file is based on Peter Shirley's book "Ray Tracing in One Weekend"
//

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <memory>

#include <GL/glew.h>

#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/bvh.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/scheduler.h>
#include <visionaray/thin_lens_camera.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

#ifdef _WIN32

//-------------------------------------------------------------------------------------------------
// https://pubs.opengroup.org/onlinepubs/007908799/xsh/drand48.html
//

double drand48()
{
    constexpr static uint64_t m = 1ULL<<48;
    constexpr static uint64_t a = 0x5DEECE66DULL;
    constexpr static uint64_t c = 0xBULL;
    thread_local static uint64_t x = 0;

    x = (a * x + c) & (m - 1ULL);

    return static_cast<double>(x) / m;
}

#endif

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
//
//

struct renderer : viewer_type
{
    using ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray \"Ray Tracing in One Weekend with CUDA\" Example")
        , bbox({ -3.0f, -3.0f, -3.0f }, { 3.0f, 3.0f, 3.0f })
        , device_sched(16, 16)
    {
        using namespace support;

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "spp",
            cl::Desc("Pixels per sample for path tracing"),
            cl::ArgRequired,
            cl::init(this->spp)
            ) );

        random_scene();

        set_background_color(vec3(0.5f, 0.7f, 1.0f));
    }

    aabb                                               bbox;
    thin_lens_camera                                   cam;
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> device_rt;
    cuda_sched<ray_type>                               device_sched;

    unsigned                                           frame_num   = 0;
    unsigned                                           spp         = 1;

    // rendering data

    index_bvh<basic_sphere<float>>                     sphere_bvh;
    std::vector<basic_sphere<float>>                   list;
    std::vector<generic_material<
            glass<float>,
            matte<float>,
            mirror<float>
            >>                                         materials;


    // copies that are located on the device
    // (we build up the initial data structures on the host!)

    cuda_index_bvh<basic_sphere<float>>                device_bvh;
    thrust::device_vector<generic_material<
            glass<float>,
            matte<float>,
            mirror<float>
            >>                                         device_materials;

    basic_sphere<float> make_sphere(vec3 center, float radius)
    {
        static int sphere_id = 0;
        basic_sphere<float> sphere(center, radius);
        sphere.prim_id = sphere_id;
        sphere.geom_id = sphere_id;
        ++sphere_id;
        return sphere;
    }

    glass<float> make_dielectric(float ior)
    {
        glass<float> mat;
        mat.ct() = from_rgb(1.0f, 1.0f, 1.0f);
        mat.kt() = 1.0f;
        mat.cr() = from_rgb(1.0f, 1.0f, 1.0f);
        mat.kr() = 1.0f;
        mat.ior() = spectrum<float>(ior);
        return mat;
    }

    matte<float> make_lambertian(vec3 cd)
    {
        matte<float> mat;
        mat.ca() = from_rgb(0.0f, 0.0f, 0.0f);
        mat.ka() = 0.0f;
        mat.cd() = from_rgb(cd);
        mat.kd() = 1.0f;
        return mat;
    }

    mirror<float> make_metal(vec3 cr)
    {
        mirror<float> mat;
        mat.cr() = from_rgb(cr);
        mat.kr() = 1.0f;
        mat.ior() = spectrum<float>(0.0);
        mat.absorption() = spectrum<float>(0.0);
        return mat;
    }

    void random_scene()
    {
        int n = 500;
        list.resize(n + 1);
        materials.resize(n + 1);
        list[0] = make_sphere(vec3(0, -1000, 0), 1000);
        materials[0] = make_lambertian(vec3(0.5f, 0.5f, 0.5f));
        int i = 1;
        for (int a = -11; a < 11; ++a)
        {
            for (int b = -11; b < 11; ++b)
            {
                float choose_mat = drand48();
                vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
                if (length(center - vec3(4, 0.2, 0)) > 0.9)
                {
                    list[i] = make_sphere(center, 0.2);
                    if (choose_mat < 0.8) // diffuse
                    {
                        materials[i] = make_lambertian(vec3(
                            static_cast<float>(drand48() * drand48()),
                            static_cast<float>(drand48() * drand48()),
                            static_cast<float>(drand48() * drand48())
                            ));
                    }
                    else if (choose_mat < 0.95) // metal
                    {
                        materials[i] = make_metal(vec3(
                            0.5f * (1.0f + static_cast<float>(drand48())),
                            0.5f * (1.0f + static_cast<float>(drand48())),
                            0.5f * (1.0f + static_cast<float>(drand48()))
                            ));
                    }
                    else
                    {
                        materials[i] = make_dielectric(1.5f);
                    }
                    ++i;
                }
            }
        }

        list[i] = make_sphere(vec3(0, 1, 0), 1.0);
        materials[i] = make_dielectric(1.5f);
        ++i;

        list[i] = make_sphere(vec3(-4, 1, 0), 1.0);
        materials[i] = make_lambertian(vec3(0.4f, 0.2f, 0.1f));
        ++i;

        list[i] = make_sphere(vec3(4, 1, 0), 1.0);
        materials[i] = make_metal(vec3(0.7f, 0.6f, 0.5f));
        ++i;

        binned_sah_builder builder;
        builder.enable_spatial_splits(true);

        sphere_bvh = builder.build(index_bvh<basic_sphere<float>>{}, list.data(), i);

        // Copy data to GPU
        device_bvh = cuda_index_bvh<basic_sphere<float>>(sphere_bvh);
        device_materials = materials;
    }


protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function, contains the rendering kernel
//

void renderer::on_display()
{
    // some setup

    pixel_sampler::basic_jittered_blend_type<float> blend_params;
    blend_params.spp = spp;
    float alpha = 1.0f / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0f - alpha;
    auto sparams = make_sched_params(
            blend_params,
            cam,
            device_rt
            );

    thrust::device_vector<cuda_index_bvh<basic_sphere<float>>::bvh_ref> device_primitives;
    device_primitives.push_back(device_bvh.ref());

    auto kparams = make_kernel_params(
            thrust::raw_pointer_cast(device_primitives.data()),
            thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
            thrust::raw_pointer_cast(device_materials.data()),
            50,
            1E-3f,
            vec4(background_color(), 1.0f),
            vec4(0.5f, 0.7f, 1.0f, 1.0f)
            );

    pathtracing::kernel<decltype(kparams)> kern;
    kern.params = kparams;

    device_sched.frame(kern, sparams);


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_FRAMEBUFFER_SRGB);

    device_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    device_rt.resize(w, h);
    device_rt.clear_color_buffer();
    frame_num = 0;

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// mouse move event
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        frame_num = 0;
        device_rt.clear_color_buffer();
    }

    viewer_type::on_mouse_move(event);
}

void renderer::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    frame_num = 0;
    device_rt.clear_color_buffer();

    viewer_type::on_space_mouse_move(event);
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
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );
    float aperture = 0.1f;
    rend.cam.set_lens_radius(aperture / 2.0f);
    rend.cam.set_focal_distance(10.0f);

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
