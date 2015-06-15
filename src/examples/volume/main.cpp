// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <memory>

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
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

using manipulators = std::vector<shared_ptr<visionaray::camera_manipulator>>;


//-------------------------------------------------------------------------------------------------
// Texture data
//

// volume data
static const float voldata[2 * 2 * 2] = {

        // slice 1
        1.0f, 0.0f,
        0.0f, 1.0f,

        // slice 2
        0.0f, 1.0f,
        1.0f, 0.0f

        };

// post-classification transfer function
static const vec4 tfdata[4 * 4] = {
        { 0.0f, 0.0f, 0.0f, 0.02f },
        { 0.7f, 0.1f, 0.2f, 0.03f },
        { 0.1f, 0.9f, 0.3f, 0.04f },
        { 1.0f, 1.0f, 1.0f, 0.05f }
        };

struct renderer
{
    using host_ray_type = basic_ray<simd::float4>;

    renderer()
        : bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
        , down_button(mouse::NoButton)
        , volume({2, 2, 2})
        , transfunc({4})
    {
        volume.set_data(voldata);
        volume.set_filter_mode(Nearest);
        volume.set_address_mode(Clamp);

        transfunc.set_data(tfdata);
        transfunc.set_filter_mode(Linear);
        transfunc.set_address_mode(Clamp);
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


    // texture references

    texture_ref<float, NormalizedFloat, 3> volume;
    texture_ref<vec4, ElementType, 1> transfunc;


};

std::unique_ptr<renderer> rend(nullptr);

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

void motion_func(int x, int y)
{
    using namespace visionaray::mouse;

    pos p = { x, y };
    for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
    {
        (*it)->handle_mouse_move( visionaray::mouse_event(mouse::Move, NoButton, p, rend->down_button, visionaray::keyboard::NoKey) );
    }
    rend->motion_pos = p;
}

void passive_motion_func(int x, int y)
{
    rend->motion_pos = { x, y };
}

void idle_func()
{
    glutPostRedisplay();
}


//-------------------------------------------------------------------------------------------------
// Display function, implements the volume rendering algorithm
//

void display_func()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = typename R::scalar_type;
    using C = vector<4, S>;

    auto sparams = make_sched_params<pixel_sampler::uniform_type>(
            rend->cam,
            rend->host_rt
            );


    // call kernel in schedulers' frame() method

    rend->host_sched.frame([&](R ray) -> result_record<S>
    {
        result_record<S> result;

        auto hit_rec = intersect(ray, rend->bbox);
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
            auto voxel = tex3D(rend->volume, tex_coord);
            C color = tex1D(rend->transfunc, voxel);

            // premultiplied alpha
            auto premult = color.xyz() * color.w;
            color = C(premult, color.w);

            // front-to-back alpha compositing
            result.color += select(
                    t < hit_rec.tfar,
                    color * (1.0f - result.color.w),
                    C(0.0)
                    );

            if ( all(result.color.w >= 0.999) )
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

    glClearColor(0.1, 0.4, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rend->host_rt.display_color_buffer();

    glutSwapBuffers();
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
    glutCreateWindow("Visionaray Volume Rendering Example");
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(passive_motion_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);

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
