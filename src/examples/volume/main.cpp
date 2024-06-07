// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/math/math.h>

#include <visionaray/texture/texture.h>

#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/cpu_buffer_rt.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Texture data
//

// volume data
VSNRAY_ALIGN(32) static const float voldata[2 * 2 * 2] = {

        // slice 1
        1.0f, 0.0f,
        0.0f, 1.0f,

        // slice 2
        0.0f, 1.0f,
        1.0f, 0.0f

        };

// post-classification transfer function
VSNRAY_ALIGN(32) static const vec4 tfdata[4 * 4] = {
        { 0.0f, 0.0f, 0.0f, 0.02f },
        { 0.7f, 0.1f, 0.2f, 0.03f },
        { 0.1f, 0.9f, 0.3f, 0.04f },
        { 1.0f, 1.0f, 1.0f, 0.05f }
        };


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float4>;

    renderer()
        : viewer_type(512, 512, "Visionaray Volume Rendering Example")
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
        , volume(array<unsigned, 3>{{2, 2, 2}})
        , transfunc(4)
    {
        volume.reset(voldata);
        volume.set_filter_mode(Nearest);
        volume.set_address_mode(Clamp);

        transfunc.reset(tfdata);
        transfunc.set_filter_mode(Linear);
        transfunc.set_address_mode(Clamp);
    }

    aabb                                        bbox;
    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED>     host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // Texture references. In this simple example, we manage
    // the texture memory ourselves, and can use texture_ref
    // as a view into the data to make use of texture interpolation,
    // texture wrap modes, etc. In general, the user will _not_
    // manage the texture memory themself. See the other examples
    // For use of the texture classes with an internal storage
    // represenation
    texture_ref<float, 3>                       volume;
    texture_ref<vec4, 1>                        transfunc;

protected:

    void on_display();
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function, implements the volume rendering algorithm
//

void renderer::on_display()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = R::scalar_type;
    using C = vector<4, S>;

    auto sparams = make_sched_params(
            cam,
            host_rt
            );


    // call kernel in schedulers' frame() method

    host_sched.frame([&](R ray) -> result_record<S>
    {
        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0);

        while ( any(t < hit_rec.tfar) )
        {
            auto pos = ray.ori + ray.dir * t;
            auto tex_coord = vector<3, S>(
                    ( pos.x + 1.0f ) / 2.0f,
                    (-pos.y + 1.0f ) / 2.0f,
                    (-pos.z + 1.0f ) / 2.0f
                    );

            // sample volume and do post-classification
            auto voxel = tex3D(volume, tex_coord);
            C color = tex1D(transfunc, voxel);

            // premultiplied alpha
            color.xyz() *= color.w;

            // front-to-back alpha compositing
            result.color += select(
                    t < hit_rec.tfar,
                    color * (1.0f - result.color.w),
                    C(0.0)
                    );

            // early-ray termination - don't traverse w/o a contribution
            if ( all(result.color.w >= 0.999f) )
            {
                break;
            }

            // step on
            t += 0.01f;
        }

        result.hit = hit_rec.hit;
        return result;
    }, sparams);


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
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

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
