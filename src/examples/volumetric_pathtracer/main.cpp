// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/math/math.h>

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/phase_function.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/random_generator.h>
#include <visionaray/sampling.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/translate_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/timer.h>
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
#if 1
        (void)pos;
        return vec3(0.0f);
#else
        // Example showing how to selectively make voxels emissive
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
#endif
    }
};


//-------------------------------------------------------------------------------------------------
//
//

struct uniform_grid
{
    template <int Mode>
    void build(::volume<Mode> const& vol, vec3i num_cells, aabb const& world_bounds)
    {
        dims = num_cells;
        majorants.resize(dims.x * size_t(dims.y) * dims.z, 0.f);
        vec3 world_size = world_bounds.size();

        // We try to use very fine sampling to make sure that
        // our majorants are a relatively tight fit
        vec3i sampling_rate(64);
        vec3i sampling_dims = dims * sampling_rate;

        auto linear_index = [this](int x, int y, int z)
        {
            return z * dims.x * size_t(dims.y) + y * dims.x + x;
        };

        auto to_world = [&](int x, int y, int z)
        {
            float fx = x / (float)(sampling_dims.x - 1);
            float fy = y / (float)(sampling_dims.y - 1);
            float fz = z / (float)(sampling_dims.z - 1);

            fx *= world_size.x;
            fy *= world_size.y;
            fz *= world_size.z;

            fx += world_bounds.min.x;
            fy += world_bounds.min.y;
            fz += world_bounds.min.z;

            return vec3(fx, fy, fz);
        };

        for (int z = 0; z < dims.z; ++z)
        {
            for (int y = 0; y < dims.y; ++y)
            {
                for (int x = 0; x < dims.x; ++x)
                {
                    for (int zz = 0; zz < sampling_rate.z; ++zz)
                    {
                        for (int yy = 0; yy < sampling_rate.y; ++yy)
                        {
                            for (int xx = 0; xx < sampling_rate.x; ++xx)
                            {
                                vec3i cell(
                                    x * sampling_rate.x + xx,
                                    y * sampling_rate.y + yy,
                                    z * sampling_rate.z + zz
                                    );
                                vec3 world_coord = to_world(cell.x, cell.y, cell.z);

                                float extinction = vol.sigma_t(world_coord);

                                size_t index = linear_index(cell.x, cell.y, cell.z);

                                majorants[index] = max(majorants[index], extinction);
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<float> majorants;
    vec3i              dims;
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

    using model_manipulators = std::vector<std::shared_ptr<model_manipulator>>;

    renderer()
        : viewer_type(512, 512, "Visionaray Volumetric Pathtracer Example")
        , bbox({ -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f })
        , host_sched(8)
    {
        model_manips.emplace_back( std::make_shared<translate_manipulator>(
                cam,
                sphere_transform,
                vec3(1.0f),
                mouse::Left
                ) );

        grid.build(vol, {16,16,16}, bbox);
    }

    aabb                                        bbox;
    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> host_rt;
    tiled_sched<host_ray_type>                  host_sched;
    uniform_grid                                grid;
    frame_counter                               counter;
    double                                      last_frame_time = 0.0;
    bool                                        print_fps = true;

    unsigned frame_num = 0;

    ::volume<1> vol;

    model_manipulators model_manips;

    mat4 sphere_transform = mat4::identity();

protected:

    void on_display();
    void on_key_press(key_event const& event);
    void on_mouse_down(visionaray::mouse_event const& event);
    void on_mouse_up(visionaray::mouse_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_space_mouse_button_press(visionaray::space_mouse_event const& event);
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
        vec3 sphere_center = (sphere_transform * vec4(0.0f, 0.0f, 0.0f, 1.0f)).xyz();
        vec3 sphere_light(30.f,70.f,20.f);

        basic_sphere<float> sph(sphere_center, 0.2f);

        result_record<float> result;

        vec3 throughput(1.0f);
        vec3 Ld(0.0f);

        auto hit_rec = intersect(r, bbox);
        auto sph_rec = intersect(r, sph);

        bool sphere_in_front = sph_rec.hit && sph_rec.t < hit_rec.tnear;
        bool sphere_hit_with_nee = false;

        if (hit_rec.hit && !sphere_in_front)
        {
            r.ori += r.dir * hit_rec.tnear;
            hit_rec.tfar -= hit_rec.tnear;

            unsigned bounce = 0;

            while (true)
            {
                vec3 Le;
                float Tr;

                float d = hit_rec.tfar;
                auto sph_rec = intersect(r, sph);
                if (sph_rec.hit)
                    d = min(d, sph_rec.t);

                collision_type coll = delta_tracking::sample_interaction(r, vol, Le, Tr, d, gen);

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

                // NEE
                if (true)
                {
                    vec3 sphere_sample = sample_surface(sph, r.ori, gen);
                    vec3 dir = normalize(sphere_sample - r.ori);
                    ray shadow_ray;
                    shadow_ray.ori = r.ori + dir * 1e-3f;
                    shadow_ray.dir = dir;
                    float d = length(sphere_sample - r.ori) - 2.f * 1e-3f;

                    // Make sure we're on the forward facing hemisphere
                    // TODO: just sample that hemisphere in the first place
                    vec3 sample_vector = normalize(sph.center + sphere_sample);
                    bool front_facing_hemisphere = dot(sample_vector, dir) < 0.f;

                    if (front_facing_hemisphere)
                    {
                        vec3 ignoreLe;
                        collision_type shadow_coll = delta_tracking::sample_interaction(
                                shadow_ray,
                                vol,
                                ignoreLe,
                                Tr,
                                d,
                                gen
                                );
                        if (front_facing_hemisphere && shadow_coll == Boundary)
                        {
                            float u1 = gen.next();
                            float u2 = gen.next();
                            vec3 n = uniform_sample_sphere(u1, u2);
                            vec3 ln = normalize(sphere_sample - sph.center);

                            auto ldotn = abs(dot(-dir, n));
                            auto ldotln = abs(dot(-dir, ln));
                            // that's for later:
                            //auto solid_angle = (ldotln * area(sph));
                            //auto lpdf = 1.0f / solid_angle;
                            Ld += sphere_light * throughput * Tr * ldotn * ldotn * ldotln;
                            sphere_hit_with_nee = true;
                        }
                    }
                }

                // Sample phase function
                vec3 scatter_dir;
                float pdf;
                f.sample(-r.dir, scatter_dir, pdf, gen);
                r.dir = scatter_dir;

                hit_rec = intersect(r, bbox);
            }
        }

        //vec3 Ld;

        // Look up sphere light
        auto sph_rec_exit = intersect(r, sph);

        if (!sphere_hit_with_nee && sph_rec_exit.hit && sph_rec_exit.t > 0.f)
        {
            Ld += sphere_light;
        }
        else
        {
            // Look up the environment
#if 1
            Ld += vec3(0.5f + 0.5f * r.dir.y);
#else
            float f = (0.5f + 0.5f * r.dir.y);
            Ld += vec3(0.5f, (1.0f - f), f);
#endif
        }
        vec3 L = Ld * throughput;

        result.color = vec4(L, 1.0f);
        result.hit = hit_rec.hit;
        return result;
    }, sparams);

    last_frame_time = counter.register_frame();

    if (print_fps)
    {
        std::cout << "FPS: " << last_frame_time << '\r';
        std::cout << std::flush;
    }

    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();

    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            manip->render();
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Keyboard handling
//

void renderer::on_key_press(key_event const& event)
{
    if (event.key() == keyboard::r)
    {
        for (auto it = model_manips.begin(); it != model_manips.end(); ++it)
        {
            if ((*it)->active())
            {
                (*it)->set_active(false);
                auto next = ++it;
                if (next != model_manips.end())
                {
                    (*next)->set_active(true);
                }
                return;
            }
        }

        (*model_manips.begin())->set_active(true);
    }

    viewer_base::on_key_press(event);
}


//-------------------------------------------------------------------------------------------------
// Mouse handling
//

void renderer::on_mouse_down(visionaray::mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_down(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_mouse_down(event);
}

void renderer::on_mouse_up(visionaray::mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_up(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_mouse_up(event);
}

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        frame_num = 0;
    }

    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_move(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_type::on_mouse_move(event);
}

void renderer::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            manip->handle_space_mouse_move(event);
            // Return unconditionally so camera
            // manipulators will not interfere
            return;
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_space_mouse_move(event);
}

void renderer::on_space_mouse_button_press(visionaray::space_mouse_event const& event)
{
    // Forward
    if (event.buttons() & space_mouse::Button1)
    {
        for (auto it = model_manips.begin(); it != model_manips.end(); ++it)
        {
            if ((*it)->active())
            {
                (*it)->set_active(false);
                auto next = ++it;
                if (next != model_manips.end())
                {
                    (*next)->set_active(true);
                }
                return;
            }
        }

        (*model_manips.begin())->set_active(true);
    }
    // Backward
    else if (event.buttons() & space_mouse::Button2)
    {
        for (auto it = model_manips.begin(); it != model_manips.end(); ++it)
        {
            if ((*it)->active())
            {
                (*it)->set_active(false);
                if (it != model_manips.begin())
                {
                    --it;
                    (*it)->set_active(true);
                    return;
                }
            }
        }

        model_manips.back()->set_active(true);
    }

    viewer_base::on_space_mouse_button_press(event);
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
