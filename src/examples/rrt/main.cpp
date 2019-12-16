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
// Texture data
//

// post-classification transfer function
VSNRAY_ALIGN(32) static const vec4 tfdata[5 * 4] = {
        { 1.0f, 0.0f, 0.0f, 1.0f   },
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 1.0f, 1.0f, 0.002f }
        };


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray Residual Ratio Tracking Example")
        //, bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , bbox({ -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f })
        , host_sched(8)
        , volume({256, 256, 256})
        , transfunc(5)
    {
        size_t width  = volume.width();
        size_t height = volume.height();
        size_t depth  = volume.depth();

        voldata.resize(width * height * depth);

        for (size_t z = 0; z < depth; ++z)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    size_t index = z * width * height + y * width + x;

                    auto xx = (float)x / width  * 4.0f - 2.0f;
                    auto yy = (float)y / height * 4.0f - 2.0f;
                    auto zz = (float)z / depth  * 4.0f - 2.0f;

                    voldata[index] = pow(xx * xx + (9.0f / 4.0f) * (yy * yy) + zz * zz - 1, 3)
                           - (xx * xx) * (zz * zz * zz) - (9.0f / 80.0f) * (yy * yy) * (zz * zz * zz);
                }
            }
        }

        volume.reset(voldata.data());
        volume.set_filter_mode(Linear);
        volume.set_address_mode(Clamp);

        transfunc.reset(tfdata);
        transfunc.set_filter_mode(Linear);
        transfunc.set_address_mode(Clamp);


        // https://graphics.pixar.com/library/ProductionVolumeRendering/paper.pdf
        // To homogenize the overall volume, we choose σn (x) in such a way that the sum
        // of all coefficients, the free-path coefficient σ ̄, becomes constant: σ ̄
        // =σa(x)+σs(x)+σn(x)=σt(x)+σn(x) (20) A consequence of the need to be constant is
        // that σ ̄ is equal or greater to the maximum of σt (x) (sometimes formulated as
        // being a majorant of σt (x)): and we can easily calculate σn (x ) from σ ̄ ≥ σt
        // (x) (21) σn (x ) = σ ̄ − σt (x).

        // So the majorant is just the maximum value (?)
        majorant = -FLT_MAX;

        for (size_t i = 0; i < voldata.size(); ++i)
        {
            majorant = max(majorant, voldata[i]);
        }
    }

    aabb                                        bbox;
    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // Volume data
    aligned_vector<float> voldata;


    // texture references

    texture_ref<float, 3>                       volume;
    texture_ref<vec4, 1>                        transfunc;

    float majorant;

    unsigned frame_num = 0;

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

    float mu_ = 50.0f;
    //float mu_ = majorant;
    float albedo = 0.8f;

    auto mu = [&](vec3 const& pos)
    {
#if 0
        const float r = 0.5f * (0.5f - fabsf (pos.y));
        const float a = (float)(M_PI * 8.0) * pos.y;
        const float dx = (cosf(a) * r - pos.x) * 2.0f;
        const float dy = (sinf(a) * r - pos.z) * 2.0f;
        return powf (fmaxf((1.0f - dx * dx - dy * dy), 0.0f), 8.0f) * mu_;
#elif 1
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
        return mu_;
#else
        vec3 tex_coord(
                ( pos.x + 1.0f ) / 2.0f,
                (-pos.y + 1.0f ) / 2.0f,
                (-pos.z + 1.0f ) / 2.0f
                );
        return tex3D(volume, tex_coord);
#endif
    };

    auto sample_interaction = [&](ray& r, float d, random_generator<float>& gen)
    {
        float t = 0.0f;
        vec3 pos;

        do
        {
            t -= log(1.0f - gen.next()) / mu_;

            pos = r.ori + r.dir * t;
            if (t >= d)
            {
                return false;
            }
        }
        while (mu(pos) < gen.next() * mu_);

        r.ori = pos;
        return true;
    };

    henyey_greenstein<float> f;
    f.g = 0.0f; // isotropic

    // call kernel in schedulers' frame() method

    host_sched.frame([&](ray r, random_generator<float>& gen) -> result_record<float>
    {
        result_record<float> result;

        float intensity = 0.0f;
        float throughput = 1.0f;

        auto hit_rec = intersect(r, bbox);

        if (hit_rec.hit)
        {
            r.ori += r.dir * hit_rec.tnear;
            hit_rec.tfar -= hit_rec.tnear;

            unsigned bounce = 0;

            while (sample_interaction(r, hit_rec.tfar, gen))
            {
                // Is the path length exceeded?
                if (bounce++ >= 1024)
                {
                    throughput = 0.0f;
                    break;
                }

                throughput *= albedo;
                // Russian roulette absorption
                if (throughput < 0.2f)
                {
                    if (gen.next() > throughput * 5.0f)
                    {
                        throughput = 0.0f;
                        break;
                    }
                    throughput = 0.2f;
                }

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
        intensity = (0.5f + 0.5f * r.dir.y) * throughput;
        vec3 L = vec3(intensity);
#else
        intensity = (0.5f + 0.5f * r.dir.y) * throughput;
        vec3 L = vec3(0.5f, (1.0f - intensity), intensity);
#endif

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
