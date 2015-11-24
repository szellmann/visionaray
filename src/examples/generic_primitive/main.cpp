// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <chrono>
#include <memory>
#include <vector>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/generic_primitive.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type     = basic_ray<simd::float4>;
    using primitive_type    = generic_primitive<basic_triangle<3, float>, basic_sphere<float>>;

    renderer()
        : viewer_type(512, 512, "Visionaray Generic Primitive Example")
        , bbox({ -1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 2.0f })
        , host_sched(8)
    {
    }

    aabb                                        bbox;
    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // rendering data

    std::vector<primitive_type>                 primitives;
    std::vector<vec3>                           normals;
    std::vector<plastic<float>>                 materials;

    void generate_frame();

protected:

    void on_display();
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Generate an animation frame
//

void renderer::generate_frame()
{
    static const size_t N = 14;

    primitives.resize(N);
    normals.resize(N);
    materials.resize(4);

    using sphere_type   = basic_sphere<float>;
    using triangle_type = basic_triangle<3, float>;


    // triangles

    // 1st pyramid

    triangle_type triangles[N];
    triangles[ 0].v1 = vec3(-1, -1,  1);
    triangles[ 0].e1 = vec3( 1, -1,  1) - triangles[ 0].v1;
    triangles[ 0].e2 = vec3( 0,  1,  0) - triangles[ 0].v1;

    triangles[ 1].v1 = vec3( 1, -1,  1);
    triangles[ 1].e1 = vec3( 1, -1, -1) - triangles[ 1].v1;
    triangles[ 1].e2 = vec3( 0,  1,  0) - triangles[ 1].v1;

    triangles[ 2].v1 = vec3( 1, -1, -1);
    triangles[ 2].e1 = vec3(-1, -1, -1) - triangles[ 2].v1;
    triangles[ 2].e2 = vec3( 0,  1,  0) - triangles[ 2].v1;

    triangles[ 3].v1 = vec3(-1, -1, -1);
    triangles[ 3].e1 = vec3(-1, -1,  1) - triangles[ 3].v1;
    triangles[ 3].e2 = vec3( 0,  1,  0) - triangles[ 3].v1;

    triangles[ 4].v1 = vec3( 1, -1,  1);
    triangles[ 4].e1 = vec3(-1, -1,  1) - triangles[ 4].v1;
    triangles[ 4].e2 = vec3( 1, -1, -1) - triangles[ 4].v1;

    triangles[ 5].v1 = vec3(-1, -1,  1);
    triangles[ 5].e1 = vec3(-1, -1, -1) - triangles[ 5].v1;
    triangles[ 5].e2 = vec3( 1, -1, -1) - triangles[ 5].v1;

    // 2nd pyramid

    triangles[ 6].v1 = vec3(0.3, 0.3, 0.7);
    triangles[ 6].e1 = vec3(0.7, 0.3, 0.7) - triangles[ 6].v1;
    triangles[ 6].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 6].v1;

    triangles[ 7].v1 = vec3(0.7, 0.3, 0.7);
    triangles[ 7].e1 = vec3(0.7, 0.3, 0.3) - triangles[ 7].v1;
    triangles[ 7].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 7].v1;

    triangles[ 8].v1 = vec3(0.7, 0.3, 0.3);
    triangles[ 8].e1 = vec3(0.3, 0.3, 0.3) - triangles[ 8].v1;
    triangles[ 8].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 8].v1;

    triangles[ 9].v1 = vec3(0.3, 0.3, 0.3);
    triangles[ 9].e1 = vec3(0.3, 0.3, 0.7) - triangles[ 9].v1;
    triangles[ 9].e2 = vec3(0.5, 0.7, 0.5) - triangles[ 9].v1;

    triangles[10].v1 = vec3(0.7, 0.3, 0.7);
    triangles[10].e1 = vec3(0.3, 0.3, 0.7) - triangles[10].v1;
    triangles[10].e2 = vec3(0.7, 0.3, 0.3) - triangles[10].v1;

    triangles[11].v1 = vec3(0.3, 0.3, 0.7);
    triangles[11].e1 = vec3(0.3, 0.3, 0.3) - triangles[11].v1;
    triangles[11].e2 = vec3(0.7, 0.3, 0.3) - triangles[11].v1;


    //
    // generate face normals assign id's:
    // prim_id links primitives and normals
    // geom_id links primitives and materials
    //

    for (size_t i = 0; i < N - 2; ++i)
    {
        triangles[i].prim_id = static_cast<unsigned>(i);
        triangles[i].geom_id = i < 6 ? 0 : 1;
        normals[i] = normalize( cross(triangles[i].e1, triangles[i].e2) );
        primitives[i] = triangles[i];
    }


    // animated spheres

    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto secs = duration_cast<milliseconds>(now.time_since_epoch()).count();

    static float y = -0.5f;
    static float m = 0.1f;
    static const int interval = 1;

    if (secs % interval == 0)
    {
        y += 0.2f * m;

        if (y < -0.5f)
        {
            m = 0.1f;
        }
        else if (y > 1.0f)
        {
            m = -0.1f;
        }
    }

    sphere_type s1;
    s1.prim_id = N - 2;
    s1.geom_id = 2;
    s1.center = vec3(-0.7f, y, 0.8);
    s1.radius = 0.5f;
    primitives[N - 2] = s1;

    sphere_type s2;
    s2.prim_id = N - 1;
    s2.geom_id = 3;
    s2.center = vec3(1.0f, 0.8, -.0f);
    s2.radius = 0.3f;
    primitives[N - 1] = s2;


    // materials

    materials[0].set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
    materials[0].set_cd( from_rgb(0.0f, 1.0f, 1.0f) );
    materials[0].set_cs( from_rgb(0.2f, 0.4f, 0.4f) );
    materials[0].set_specular_exp( 16.0f );

    materials[1].set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
    materials[1].set_cd( from_rgb(1.0f, 0.0f, 0.0f) );
    materials[1].set_cs( from_rgb(0.5f, 0.2f, 0.2f) );
    materials[1].set_specular_exp( 128.0f );

    materials[2].set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
    materials[2].set_cd( from_rgb(0.0f, 0.0f, 1.0f) );
    materials[2].set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
    materials[2].set_specular_exp( 128.0f );

    materials[3].set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
    materials[3].set_cd( from_rgb(1.0f, 1.0f, 1.0f) );
    materials[3].set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
    materials[3].set_specular_exp( 32.0f );

    for (auto& m : materials)
    {
        m.set_ka( 1.0f );
        m.set_kd( 1.0f );
        m.set_ks( 1.0f );
    }
}


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

    auto sparams = make_sched_params(
            pixel_sampler::uniform_type{},
            cam,
            host_rt
            );



    // a light positioned slightly distant above the scene
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position( vec3(0.0f, 10.0f, 10.0f) );

    std::vector<point_light<float>> lights;
    lights.push_back(light);


    generate_frame();

    auto kparams = make_kernel_params(
            primitives.data(),
            primitives.data() + primitives.size(),
            normals.data(),
            materials.data(),
            lights.data(),
            lights.data() + lights.size(),
            4,                          // number of reflective bounces
            0.0001                      // epsilon to avoid self intersection by secondary rays
            );

    auto kernel = whitted::kernel<decltype(kparams)>();
    kernel.params = kparams;

    host_sched.frame(kernel, sparams);

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
    catch (std::exception& e)
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
