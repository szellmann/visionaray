// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <chrono>
#include <iostream>
#include <memory>
#include <ostream>
#include <thread>
#include <vector>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/math/io.h>

#include <visionaray/texture/texture_traits.h>

#include <visionaray/bvh.h>
#include <visionaray/generic_primitive.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/bvh_outline_renderer.h>
#include <common/cpu_buffer_rt.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

using namespace visionaray;

namespace juggler
{
using scalar_type = simd::float4;
}
using viewer_type = viewer_glut;


namespace visionaray
{

struct procedural_texture
{
    bool checker = false;

    // texture size
    int w = 512;
    int h = 512;

    // checker size (relative to texture size)
    int cw = 1;
    int ch = 1;
};

vec4 tex2D(procedural_texture const& tex, vec2 const& coord)
{
    if (tex.checker)
    {
        vec3 cartesian(
                sin(coord.x) * cos(coord.y),
                sin(coord.x) * sin(coord.y),
                cos(coord.x)
                );

        int x = ((cartesian.x + 0.5f) / 2.0f * (tex.w - tex.cw * 2)) / tex.cw;
        int y = ((cartesian.z + 0.5f) / 2.0f * (tex.h - tex.ch * 2)) / tex.ch;

        if (x % 2 != y % 2)
        {
            return vec4(0.0f, 1.0f, 0.0f, 1.0f);
        }
        else
        {
            return vec4(1.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    else
    {
        return vec4(1.0f);
    }
}

template <>
struct texture_dimensions<procedural_texture>
{
    enum { value = 2 };
};

} // visionaray

//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type  = basic_ray<juggler::scalar_type>;
    using primitive_type = basic_sphere<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray Juggler Example")
        , bbox({ -1.0f, 0.0f, 0.0f }, { 1.0f, 3.0f, 2.0f })
        , host_sched(8)
        , pool(std::thread::hardware_concurrency())
    {
        build_scene();
    }

    aabb                                        bbox;
    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // rendering data

    aligned_vector<basic_sphere<float>>         primitives;
    index_bvh<basic_sphere<float>>              bvh;
    aligned_vector<generic_material<plastic<float>, mirror<float>>> materials;
    aligned_vector<procedural_texture>          textures;

    // scene data

    float foot_radius = 0.05f;
    float knee_radius = 0.1f;
    float hip_radius = 0.33f;
    float shoulder_radius = 0.38f;
    float ellbow_radius = 0.1f;
    float hand_radius = 0.05f;
    float neck_radius = 0.1f;
    float head_radius = 0.23f;
    float eye_radius = 0.05f;
    float ball_radius = 0.28f;

    basic_sphere<float>                 ground;
    aligned_vector<basic_sphere<float>> shank;
    aligned_vector<basic_sphere<float>> thigh;
    aligned_vector<basic_sphere<float>> torso;
    aligned_vector<basic_sphere<float>> upper_arm;
    aligned_vector<basic_sphere<float>> forearm;
    basic_sphere<float>                 neck;
    basic_sphere<float>                 head;
    basic_sphere<float>                 hair;
    basic_sphere<float>                 eye;
    basic_sphere<float>                 balls[3];

    unsigned tick1 = 0;
    unsigned tick2 = 0;
    unsigned num_cycles = 0;

    bool show_bvh = false;

    bvh_outline_renderer                outlines;

    thread_pool pool;

    void build_scene();

    void generate_frame(float t);

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Build up the scene geometry
//

void renderer::build_scene()
{
    enum geom_ids
    {
        GroundId, SkinId, TorsoId, HairId, EyeId, BallId, MaxId
    };

    materials.resize(MaxId);

    textures.resize(MaxId);
    textures[GroundId].checker = true;

    plastic<float> ground_mat;
    ground_mat.ca() = from_rgb(0.7f, 0.7f, 0.7f);
    ground_mat.cd() = from_rgb(1.0f, 1.0f, 1.0f);
    ground_mat.cs() = from_rgb(0.0f, 0.0f, 0.0f);
    ground_mat.ka() = 1.0f;
    ground_mat.kd() = 1.0f;
    ground_mat.ks() = 0.0f;
    ground_mat.specular_exp() = 0.0f;
    materials[GroundId] = ground_mat;

    plastic<float> skin_mat;
    skin_mat.ca() = from_rgb(0.4f, 0.3f, 0.0f);
    skin_mat.cd() = from_rgb(1.0f, 0.7f, 0.7f);
    skin_mat.cs() = from_rgb(1.0f, 1.0f, 1.0f);
    skin_mat.ka() = 1.0f;
    skin_mat.kd() = 1.0f;
    skin_mat.ks() = 1.0f;
    skin_mat.specular_exp() = 512.0f;
    materials[SkinId] = skin_mat;

    plastic<float> torso_mat;
    torso_mat.ca() = from_rgb(0.5f, 0.0f, 0.0f);
    torso_mat.cd() = from_rgb(0.8f, 0.0f, 0.0f);
    torso_mat.cs() = from_rgb(1.0f, 1.0f, 1.0f);
    torso_mat.ka() = 1.0f;
    torso_mat.kd() = 1.0f;
    torso_mat.ks() = 1.0f;
    torso_mat.specular_exp() = 512.0f;
    materials[TorsoId] = torso_mat;

    plastic<float> hair_mat;
    hair_mat.ca() = from_rgb(0.0f, 0.0f, 0.0f);
    hair_mat.cd() = from_rgb(0.2f, 0.0f, 0.0f);
    hair_mat.cs() = from_rgb(1.0f, 1.0f, 1.0f);
    hair_mat.ka() = 1.0f;
    hair_mat.kd() = 1.0f;
    hair_mat.ks() = 1.0f;
    hair_mat.specular_exp() = 512.0f;
    materials[HairId] = hair_mat;

    plastic<float> eye_mat;
    eye_mat.ca() = from_rgb(0.0f, 0.0f, 0.2f);
    eye_mat.cd() = from_rgb(0.0f, 0.0f, 0.6f);
    eye_mat.cs() = from_rgb(1.0f, 1.0f, 1.0f);
    eye_mat.ka() = 1.0f;
    eye_mat.kd() = 1.0f;
    eye_mat.ks() = 1.0f;
    eye_mat.specular_exp() = 512.0f;
    materials[EyeId] = eye_mat;

    mirror<float> ball_mat;
    ball_mat.cr() = from_rgb(1.0f, 1.0f, 1.0f);
    ball_mat.kr() = 1.0f;
    ball_mat.ior() = spectrum<float>(1.0);
    ball_mat.absorption() = spectrum<float>(0.3);
    materials[BallId] = ball_mat;


    // Ground is just one huge sphere
    ground.center = vec3f(0.0f, -1000.0f, 0.0f);
    ground.radius = 1000.0f;
    ground.geom_id = GroundId;

    // Knee is part of the shank
    shank.resize(10);
    for (size_t i = 0; i < shank.size(); ++i)
    {
        float alpha = i / static_cast<float>(shank.size() - 1);
        shank[i].center = vec3f(0.0f, foot_radius * 2.0f * i, 0.0f);
        shank[i].radius = lerp_r(foot_radius, knee_radius, alpha);
        shank[i].geom_id = SkinId;
    }

    // Thigh; all the spheres are "knee-sized"
    thigh.resize(6);
    for (size_t i = 0; i < thigh.size(); ++i)
    {
        thigh[i].center = vec3f(0.0f, knee_radius * i, 0.0f);
        thigh[i].radius = knee_radius;
        thigh[i].geom_id = SkinId;
    }

    // Torso; a couple of red spheres
    torso.resize(7);
    for (size_t i = 0; i < torso.size(); ++i)
    {
        float alpha = i / static_cast<float>(torso.size() - 1);
        torso[i].center = vec3f(0.0f, 0.07f * i, 0.0f);
        torso[i].radius = lerp_r(hip_radius, shoulder_radius, alpha);
        torso[i].geom_id = TorsoId;
    }

    // Upper arm; spheres anchored at the shoulde
    // Ellbow is part of upper arm
    upper_arm.resize(8);
    for (size_t i = 0; i < upper_arm.size(); ++i)
    {
        upper_arm[i].center = vec3f(0.0f, -0.09f * i, 0.0f);
        upper_arm[i].radius = ellbow_radius;
        upper_arm[i].geom_id = SkinId;
    }

    // Forearm; anchoreed at ellbow
    forearm.resize(10);
    for (size_t i = 0; i < forearm.size(); ++i)
    {
        float alpha = i / static_cast<float>(forearm.size() - 1);
        forearm[i].center = vec3f(0.0f, -0.09f * i, 0.0f);
        forearm[i].radius = lerp_r(ellbow_radius, hand_radius, alpha);
        forearm[i].geom_id = SkinId;
    }

    // Spheres for neck, head, hair and eyes
    neck.center = vec3f(0.0f, 0.0f, 0.0f);
    neck.radius = neck_radius;
    neck.geom_id = SkinId;

    head.center = vec3f(0.0f, 0.0f, 0.0f);
    head.radius = head_radius;
    head.geom_id = SkinId;

    hair.center = vec3f(0.0f, 0.0f, 0.0f);
    hair.radius = head_radius;
    hair.geom_id = HairId;

    eye.center = vec3f(0.0f, 0.0f, 0.0f);
    eye.radius = eye_radius;
    eye.geom_id = EyeId;

    // The balls
    for (int i = 0; i < 3; ++i)
    {
        balls[i].center = vec3f(0.0f, 0.0f, 0.0f);
        balls[i].radius = ball_radius;
        balls[i].geom_id = BallId;
    }
}


//-------------------------------------------------------------------------------------------------
// Generate an animation frame
//

void renderer::generate_frame(float t)
{
    vec3 knee_pos[2];
    vec3 hip_pos[2];
    vec3 shoulder_pos[2];
    vec3 ellbow_pos[2];

    primitives.clear();

    // The earth
    primitives.push_back(ground);

    // Shanks
    for (int i = 0; i < 2; ++i)
    {
        int sgn = i * 2 - 1;

        // Move apart
        mat4 trans = mat4::translation(0.3f * sgn, 0.0f, 0.0f);

        // Outward motion of knee, center of rotation is the foot
        float zrot = t < 0.5f ? lerp_r(3.0f, 5.0f, t)
                              : lerp_r(5.0f, 3.0f, t);
        trans = trans * mat4::rotation(vec3f(0.0f, 0.0f, 1.0f), -zrot * sgn * constants::degrees_to_radians<float>());

        // Forward motion of knee, center of rotation is the foot
        float xrot = t < 0.5f ? lerp_r(10.0f, 25.0f, t)
                              : lerp_r(25.0f, 10.0f, t);
        trans = trans * mat4::rotation(vec3f(1.0f, 0.0f, 0.0f), xrot * constants::degrees_to_radians<float>());

        for (auto const& sph : shank)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            primitives.push_back(tsph);
        }

        // Remember the transformed knee positions
        knee_pos[i] = primitives.back().center;
    }

    // Thighs
    for (int i = 0; i < 2; ++i)
    {
        int sgn = i * 2 - 1;

        // We'll rotate the thighs around the knees; as the knees
        // are part of the shanks, we'll move the thighs up a bit
        // to rotate around a knee that's not really there
        mat4 trans = mat4::translation(0.0f, thigh[0].radius, 0.0f);

        // Inward motion of hips, center of rotation is the knee
        float zrot = 10.0f;
        trans = trans * mat4::rotation(vec3f(0.0f, 0.0f, 1.0f), zrot * sgn * constants::degrees_to_radians<float>());

        // Backward motion of hips, center of rotation is the knee
        float xrot = t < 0.5f ? lerp_r(0.0f, -50.0f, t)
                              : lerp_r(-50.0f, 0.0f, t);
        trans = trans * mat4::rotation(vec3f(1.0f, 0.0f, 0.0f), xrot * constants::degrees_to_radians<float>());

        // Perform rotations around knee
        aligned_vector<basic_sphere<float>> thigh_rot;
        for (auto const& sph : thigh)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            thigh_rot.push_back(tsph);
        }

        // Finally move the whole thigh up to above the knee
        trans = mat4::translation(knee_pos[i]);
        for (auto const& sph : thigh_rot)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            primitives.push_back(tsph);
        }

        // Remember the transformed hip positions
        hip_pos[i] = primitives.back().center;
    }

    // Torso
    {
        mat4 trans = mat4::translation(
                (hip_pos[0] + hip_pos[1]) / 2.0f + vec3f(0.0f, hip_radius / 2.0f,
                0.0f)
                );

        // Move the torso to above the knees, keep upright all the time
        for (auto const& sph : torso)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            primitives.push_back(tsph);
        }

        // Remember the transformed hip positions
        float r = primitives.back().radius;
        shoulder_pos[0] = primitives.back().center + vec3(-(r - 0.05f), 0.25f, 0.0f);
        shoulder_pos[1] = primitives.back().center + vec3( (r - 0.05f), 0.25f, 0.0f);
    }

    // Upper arms
    for (int i = 0; i < 2; ++i)
    {
        int sgn = i * 2 - 1;

        mat4 trans = mat4::identity();

        // Outward motion of upper arm, center of rotation is the shoulder
        float zrot = t < 0.5f ? lerp_r(20.0f, 40.0f, t)
                              : lerp_r(40.0f, 20.0f, t);
        trans = trans * mat4::rotation(vec3f(0.0f, 0.0f, 1.0f), zrot * sgn * constants::degrees_to_radians<float>());

        // Backward motion of upper arm, center of rotation is the shoulder
        float xrot = t < 0.5f ? lerp_r(10.0f, 30.0f, t)
                              : lerp_r(30.0f, 10.0f, t);
        trans = trans * mat4::rotation(vec3f(1.0f, 0.0f, 0.0f), xrot * constants::degrees_to_radians<float>());

        // Perform rotations around knee
        aligned_vector<basic_sphere<float>> upper_arm_rot;
        for (auto const& sph : upper_arm)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            upper_arm_rot.push_back(tsph);
        }

        // Align with shoulders
        trans = mat4::translation(shoulder_pos[i]);
        for (auto const& sph : upper_arm_rot)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            primitives.push_back(tsph);
        }

        // Remember the transformed ellbow positions
        ellbow_pos[i] = primitives.back().center;
    }

    // Forearms
    for (int i = 0; i < 2; ++i)
    {
        int sgn = i * 2 - 1;

        // Move a bit down as ellbow is part of upper arm
        mat4 trans = mat4::translation(0.02f * sgn, -ellbow_radius, 0.0f);

        // Outward motion of forearm, center of rotation is the ellbow
        float zrot = t < 0.5f ? lerp_r(0.0f, 10.0f, t)
                              : lerp_r(10.0f, 0.0f, t);
        trans = trans * mat4::rotation(vec3f(0.0f, 0.0f, 1.0f), zrot * sgn * constants::degrees_to_radians<float>());

        // Upward motion of forearm, center of rotation is the ellbow
        float xrot = t < 0.5f ? lerp_r(-60.0f, -110.0f, t)
                              : lerp_r(-110.0f, -60.0f, t);
        trans = trans * mat4::rotation(vec3f(1.0f, 0.0f, 0.0f), xrot * constants::degrees_to_radians<float>());

        // Perform rotations around knee
        aligned_vector<basic_sphere<float>> forearm_rot;
        for (auto const& sph : forearm)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            forearm_rot.push_back(tsph);
        }

        // Align with ellbows
        trans = mat4::translation(ellbow_pos[i]);
        for (auto const& sph : forearm_rot)
        {
            basic_sphere<float> tsph = sph;
            tsph.center = (trans * vec4(tsph.center, 1.0f)).xyz();
            primitives.push_back(tsph);
        }
    }

    // Neck, head, hair, and eyes
    {
        // TODO: use translations?!
        neck.center = vec3(
                0.0f,
                shoulder_pos[0].y + neck_radius * 1.5f,
                shoulder_pos[0].z
                );
        primitives.push_back(neck);

        head.center = vec3(
                0.0f,
                neck.center.y + (neck.radius / 2.0f) + head.radius,
                shoulder_pos[0].z);
        primitives.push_back(head);

        hair.center = head.center + vec3(0.0f, 0.01f, -0.01f);
        primitives.push_back(hair);

        eye.center = head.center + vec3(-0.1f, 0.05f, head.radius - 0.01f);
        primitives.push_back(eye);

        eye.center = head.center + vec3( 0.1f, 0.05f, head.radius - 0.01f);
        primitives.push_back(eye);
    }

    vec3 start_pos(0.6f, 1.6f, 0.8f);

    // 1st ball, moves along small parabola
    if (1)
    {
        float tt = t + 0.5f;
        if (tt > 1.0f)
        {
            tt -= 1.0f;
        }
        float x = lerp_r(-1.0f, 1.0f, tt);
        float fx =  -0.4f * (x * x) + 0.4f;

        balls[0].center = start_pos;
        balls[0].center.x *= x;
        balls[0].center.y += fx;
        primitives.push_back(balls[0]);
    }

    // 2nd ball, moves along large parabola,
    // with t offset by 1/3
    if (1)
    {
        float tc = num_cycles % 2;

        float tt = tc + t + 0.5f;
        if (tt > 2.0f)
        {
            tt -= 2.0f;
        }
        float x = lerp_r(1.0f, -1.0f, tt/2);//std::cout << x << '\n';
        float fx =  -1.8f * (x * x) + 1.8f;

        balls[1].center = start_pos;
        balls[1].center.x *= x;
        balls[1].center.y += fx;
        primitives.push_back(balls[1]);
    }

    // 2nd ball, moves along large parabola,
    // with t offset by 1/3
    if (1)
    {
        float tc = (num_cycles + 1) % 2;

        float tt = tc + t + 0.5f;
        if (tt > 2.0f)
        {
            tt -= 2.0f;
        }
        float x = lerp_r(1.0f, -1.0f, tt/2);//std::cout << x << '\n';
        float fx =  -1.8f * (x * x) + 1.8f;

        balls[1].center = start_pos;
        balls[1].center.x *= x;
        balls[1].center.y += fx;
        primitives.push_back(balls[1]);
    }


    // Finally assign primitive ids
    for (unsigned i = 0; i < primitives.size(); ++i)
    {
        primitives[i].prim_id = i;
    }

    if (bvh.num_nodes() == 0)
    {
        lbvh_builder builder;
        bvh = builder.build(index_bvh<basic_sphere<float>>{}, primitives.data(), primitives.size());
    }
    else
    {
        bvh_refitter refitter;
        refitter.refit(bvh, primitives.data(), primitives.size(), pool);
    }

    outlines.init(bvh);
}


//-------------------------------------------------------------------------------------------------
// Window close function
//

void renderer::on_close()
{
    outlines.destroy();
}


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

    pixel_sampler::uniform_type ps;
    ps.ssaa_factor = 4; // 4x SSAA

    auto sparams = make_sched_params(ps, cam, host_rt);


    // a light positioned slightly distant above the scene
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position( vec3(0.0f, 10.0f, 10.0f) );
    light.set_constant_attenuation(1.0f);
    light.set_linear_attenuation(0.0f);
    light.set_quadratic_attenuation(0.0f);

    std::vector<point_light<float>> lights;
    lights.push_back(light);


    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();

    tick2 = ms % 1000;
    if (tick2 < tick1)
    {
        ++num_cycles;
    }
    tick1 = tick2;

    generate_frame((ms % 1000) / 1000.0f);

    vec3* dummies = nullptr;
    aligned_vector<vec2> tex_coords(1);

    aligned_vector<index_bvh<basic_sphere<float>>::bvh_ref> refs;
    refs.push_back(bvh.ref());

    auto kparams = make_kernel_params(
            normal_binding{},           // has no normal binding
            refs.data(),
            refs.data() + refs.size(),
            dummies,                    // has no normals
            dummies,                    // has no normals
            tex_coords.data(),          // has no tex coords (but may still not be null (TODO!!!))
            materials.data(),
            textures.data(),
            lights.data(),
            lights.data() + lights.size(),
            4,                          // number of reflective bounces
            0.001f,                     // epsilon to avoid self intersection by secondary rays
            vec4(background_color(), 1.0f),
            vec4(0.6f)
            );

    whitted::kernel<decltype(kparams)> kernel;
    kernel.params = kparams;

    host_sched.frame(kernel, sparams);

    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();

    if (show_bvh)
    {
        outlines.frame(cam.get_view_matrix(), cam.get_proj_matrix());
    }
}


//-------------------------------------------------------------------------------------------------
// Key press event
//

void renderer::on_key_press(key_event const& event)
{
    if (event.key() == 'b')
    {
        show_bvh = !show_bvh;
    }

    viewer_type::on_key_press(event);
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
