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

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/phase_function.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/random_generator.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Procedural volumes
// Using the same as in "Ray Tracing Inhomogeneous Volumes" by Matthias Raab (Ray Tracing Gems 1)
//

template <int Mode>
struct volume
{
    // Extinction (absorption+scattering)
    float sigma_t(vec3 const& pos) const
    {
        if (Mode == 0) // spiral
        {
            const float r = 0.5f * (0.5f - fabsf (pos.y));
            const float a = (float)(M_PI * 8.0) * pos.y;
            const float dx = (cosf(a) * r - pos.x) * 2.0f;
            const float dy = (sinf(a) * r - pos.z) * 2.0f;
            return powf (fmaxf((1.0f - dx * dx - dy * dy), 0.0f), 8.0f) * sigma_();
        }
        else // menger sponge
        {
            vec3 ppos = pos + vec3(0.5f, 0.5f, 0.5f);
            const unsigned int steps = 3;
            for (unsigned int i = 0; i < steps; ++i)
            {
                ppos *= 3.0f;

                const int s = ((int)ppos.x & 1) + ((int)ppos.y & 1) + ((int)ppos.z & 1);

                if (s >= 2)
                {
                    return 0.0f;
                }
            }
            return sigma_();
        }
    }

    float sigma_a(vec3 const& pos) const
    {
        // Model extinction as 1:1 absorption and out-scattering
        return sigma_t(pos) / 2.0f;
    }

    // Majorant
    float sigma_() const
    {
        return 50.0f;
    }

    // Emission
    vec3 Le(vec3 const& pos) const
    {
        if (pos.x > 0.16f && pos.y > 0.16f && pos.z > 0.16f)
         //&& pos.x < 0.33f && pos.y < 0.33f && pos.z < 0.33f)
        {
            return vec3(4.0f, 2.0f, 1.5f);
            //return vec3(120.0f, 60.0f, 45.0f);
        }
        else
        {
            return vec3(0.0f);
        }
    }
};


//-------------------------------------------------------------------------------------------------
//
//

enum collision_type
{
    Scattering,
    Emission,
    Boundary,
};

namespace delta_tracking
{

template <int Mode>
collision_type sample_interaction(ray& r, ::volume<Mode> const& vol, vec3& Le, float& Tr,float d, random_generator<float>& gen)
{
    collision_type result;

    float t = 0.0f;
    vec3 pos;

    Le = vec3(0.0f);

    while (true)
    {
        t -= log(1.0f - gen.next()) / vol.sigma_();

        pos = r.ori + r.dir * t;
        if (t >= d)
        {
            result = Boundary;
            break;
        }

        float u2 = gen.next();
        if (u2 < vol.sigma_a(pos) / vol.sigma_t(pos))
        {
            Le = vol.Le(pos);
            result = Emission;
            break;
        }
        else if (vol.sigma_t(pos) >= u2 * vol.sigma_())
        {
            result = Scattering;
            break;
        }
    }

    Tr = result == Boundary ? 1.0f : 0.0f;
    r.ori = pos;

    return result;
}

}


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray Residual Ratio Tracking Example")
        , bbox({ -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f })
        , host_sched(8)
    {
    }

    aabb                                        bbox;
    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    unsigned frame_num = 0;

    ::volume<1> vol;

protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function, implements the volume rendering algorithm
//

void renderer::on_display()
{
    // some setup

    pixel_sampler::basic_jittered_blend_type<float> blend_params;
    float alpha = 1.0f / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0f - alpha;

    auto sparams = make_sched_params(
            blend_params,
            cam,
            host_rt
            );

    float albedo = 0.8f;

    henyey_greenstein<float> f;
    f.g = 0.0f; // isotropic

    // call kernel in schedulers' frame() method

    host_sched.frame([&](ray r, random_generator<float>& gen) -> result_record<float>
    {
        result_record<float> result;

        vec3 throughput(1.0f);

        auto hit_rec = intersect(r, bbox);

        if (hit_rec.hit)
        {
            r.ori += r.dir * hit_rec.tnear;
            hit_rec.tfar -= hit_rec.tnear;

            unsigned bounce = 0;

            while (true)
            {
                vec3 Le;
                float Tr;
                collision_type coll = delta_tracking::sample_interaction(r, vol, Le, Tr, hit_rec.tfar, gen);

                if (coll == Boundary)
                {
                    // Outside of the volume
                    break;
                }

                // Is the path length exceeded?
                if (bounce++ >= 1024)
                {
                    throughput = vec3(0.0f);
                    break;
                }

                throughput *= albedo;
                // Russian roulette absorption
                float prob = max_element(throughput);
                if (prob < 0.2f)
                {
                    if (gen.next() > prob)
                    {
                        throughput = vec3(0.0f);
                        break;
                    }
                    throughput /= prob;
                }

                throughput += Le;

                // Sample phase function
                vec3 scatter_dir;
                float pdf;
                f.sample(-r.dir, scatter_dir, pdf, gen);
                r.dir = scatter_dir;

                hit_rec = intersect(r, bbox);
            }
        }

        // Look up the environment
#if 1
        vec3 Ld(0.5f + 0.5f * r.dir.y);
#else
        float f = (0.5f + 0.5f * r.dir.y);
        vec3 Ld(0.5f, (1.0f - f), f);
#endif
        vec3 L = Ld * throughput;

        result.color = vec4(L, 1.0f);
        result.hit = hit_rec.hit;
        return result;
    }, sparams);


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}


void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        frame_num = 0;
    }

    viewer_type::on_mouse_move(event);
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
