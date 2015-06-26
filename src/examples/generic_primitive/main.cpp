// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <chrono>
#include <memory>
#include <vector>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#include <GL/gl.h>
#include <GL/glut.h>

#endif

#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif

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

using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

using manipulators = std::vector<shared_ptr<visionaray::camera_manipulator>>;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer
{
    using host_ray_type     = basic_ray<simd::float4>;
    using tex_ref           = texture_ref<unorm<8>, NormalizedFloat, 2>;
    using primitive_type    = generic_primitive<basic_triangle<3, float>, basic_sphere<float>>;

    renderer()
        : bbox({ -1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 2.0f })
        , host_sched(8)
        , down_button(mouse::NoButton)
    {
    }

    aabb                                        bbox;
    camera                                      cam;
    manipulators                                manips;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    mouse::button down_button;
    mouse::pos motion_pos;
    mouse::pos down_pos;
    mouse::pos up_pos;


    // rendering data

    std::vector<primitive_type>                 primitives;
    std::vector<vec3>                           normals;
    std::vector<plastic<float>>                 materials;

    void generate_frame();
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

std::unique_ptr<renderer> rend(nullptr);


//-------------------------------------------------------------------------------------------------
// Display function
//

void display_func()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = typename R::scalar_type;
    using V = vector<3, S>;

    auto sparams = make_sched_params<pixel_sampler::uniform_type>(
            rend->cam,
            rend->host_rt
            );



    // a light positioned slightly distant above the scene
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position( vec3(0.0f, 10.0f, 10.0f) );

    std::vector<point_light<float>> lights;
    lights.push_back(light);


    rend->generate_frame();

    auto kparams = make_params<normals_per_face_binding>(
            rend->primitives.data(),
            rend->primitives.data() + rend->primitives.size(),
            rend->normals.data(),
            rend->materials.data(),
            lights.data(),
            lights.data() + lights.size(),
            4,                          // number of reflective bounces
            0.0001,                     // epsilon to avoid self intersection by secondary rays
            vec4(0.1, 0.4, 1.0, 1.0)    // background color
            );

    auto kernel = whitted::kernel<decltype(kparams)>();
    kernel.params = kparams;

    rend->host_sched.frame(kernel, sparams);

    // display the rendered image

    glClearColor(0.1, 0.4, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rend->host_rt.display_color_buffer();

    glutSwapBuffers();
}


//-------------------------------------------------------------------------------------------------
// on idle, refresh the viewport
//

void idle_func()
{
    glutPostRedisplay();
}


//-------------------------------------------------------------------------------------------------
// mouse handling
//

void motion_func(int x, int y)
{
    using namespace visionaray::mouse;

    pos p = { x, y };
    for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
    {
        (*it)->handle_mouse_move( visionaray::mouse_event(
                mouse::Move,
                NoButton,
                p,
                rend->down_button,
                visionaray::keyboard::NoKey
                ) );
    }
    rend->motion_pos = p;
}

void mouse_func(int button, int state, int x, int y)
{
    using namespace visionaray::mouse;

    mouse::button b = map_glut_button(button);
    pos p = { x, y };

    if (state == GLUT_DOWN)
    {
        for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
        {
            (*it)->handle_mouse_down(visionaray::mouse_event(mouse::ButtonDown, b, p));
        }
        rend->down_pos = p;
        rend->down_button = b;
    }
    else if (state == GLUT_UP)
    {
        for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
        {
            (*it)->handle_mouse_up(visionaray::mouse_event(mouse::ButtonUp, b, p));
        }
        rend->up_pos = p;
        rend->down_button = mouse::NoButton;
    }
}

void passive_motion_func(int x, int y)
{
    rend->motion_pos = { x, y };
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    rend = std::unique_ptr<renderer>(new renderer);

    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(512, 512);
    glutCreateWindow("Visionaray Generic Primitive Example");
    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutMotionFunc(motion_func);
    glutMouseFunc(mouse_func);
    glutPassiveMotionFunc(passive_motion_func);

    glewInit();

    float aspect = 1.0f;

    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->cam.set_viewport(0, 0, 512, 512);
    rend->cam.view_all( rend->bbox );

    rend->manips.push_back( make_shared<visionaray::arcball_manipulator>(rend->cam, mouse::Left) );
    rend->manips.push_back( make_shared<visionaray::pan_manipulator>(rend->cam, mouse::Middle) );
    rend->manips.push_back( make_shared<visionaray::zoom_manipulator>(rend->cam, mouse::Right) );

    glViewport(0, 0, 512, 512);
    rend->host_rt.resize(512, 512);

    glutMainLoop();
}
