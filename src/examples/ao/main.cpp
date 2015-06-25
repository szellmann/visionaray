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

#include <visionaray/detail/traverse.h>
#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/scheduler.h>
#include <visionaray/surface.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

#include <common/model.h>
#include <common/obj_loader.h>

using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

using manipulators = std::vector<shared_ptr<visionaray::camera_manipulator>>;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer
{
    using host_ray_type = basic_ray<simd::float4>;

    renderer()
        : host_sched(8)
        , down_button(mouse::NoButton)
    {
    }

    camera                                      cam;
    manipulators                                manips;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    model mod;
    index_bvh<model::triangle_list::value_type> host_bvh;
    unsigned                                    frame_num       = 0;

    mouse::button down_button;
    mouse::pos motion_pos;
    mouse::pos down_pos;
    mouse::pos up_pos;
};

std::unique_ptr<renderer> rend(nullptr);


//-------------------------------------------------------------------------------------------------
// Display function, implements the AO kernel
//

void display_func()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = typename R::scalar_type;
    using C = vector<4, S>;
    using V = vector<3, S>;

    auto sparams = make_sched_params<pixel_sampler::jittered_blend_type>(
            rend->cam,
            rend->host_rt
            );


    using bvh_ref = index_bvh<model::triangle_list::value_type>::bvh_ref;

    std::vector<bvh_ref> bvhs;
    bvhs.push_back(rend->host_bvh.ref());

    auto prims_begin = bvhs.data();
    auto prims_end   = bvhs.data() + bvhs.size();

    rend->host_sched.frame([&](R ray, sampler<S>& samp) -> result_record<S>
    {
        result_record<S> result;
        result.color = C(0.1, 0.4, 1.0, 1.0);

        auto hit_rec = closest_hit(
                ray,
                prims_begin,
                prims_end
                );

        result.hit = hit_rec.hit;

        if (any(hit_rec.hit))
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;
            result.isect_pos  = hit_rec.isect_pos;

            C clr(1.0);

            auto n = get_normal(rend->mod.normals.data(), hit_rec);

            // Make an ortho basis (TODO: move to library)
            auto w = n;
            auto v = select(
                    abs(w.x) > abs(w.y),
                    normalize( V(-w.z, S(0.0), w.x) ),
                    normalize( V(S(0.0), w.z, -w.y) )
                    );
            auto u = cross(v, w);

            static const int AO_Samples = 8;
            S radius(0.1);


            for (int i = 0; i < AO_Samples; ++i)
            {
                auto sp = cosine_sample_hemisphere(samp.next(), samp.next());

                auto dir = normalize( sp.x * u + sp.y * v + sp.z * w );

                R ao_ray;
                ao_ray.ori = hit_rec.isect_pos + dir * S(1E-3f);
                ao_ray.dir = dir;

                auto ao_rec = any_hit(
                        ao_ray,
                        prims_begin,
                        prims_end,
                        typename R::scalar_type(radius)
                        );

                clr = select(
                        ao_rec.hit,
                        clr - S(1.0f / AO_Samples),
                        clr
                        );
            }

            result.color      = select( hit_rec.hit, C(clr.xyz(), S(1.0)), result.color );

        }

        return result;
    }, sparams, ++rend->frame_num);


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

    rend->frame_num = 0;

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
    glutCreateWindow("Visionaray Ambient Occlusion Example");
    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutMotionFunc(motion_func);
    glutMouseFunc(mouse_func);
    glutPassiveMotionFunc(passive_motion_func);

    glewInit();

    try
    {
        visionaray::load_obj(argv[1], rend->mod);
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Creating BVH...\n";

    rend->host_bvh = build<index_bvh<model::triangle_list::value_type>>(
            rend->mod.primitives.data(),
            rend->mod.primitives.size()
            );

    std::cout << "Ready\n";

    float aspect = 1.0f;

    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->cam.set_viewport(0, 0, 512, 512);
    rend->cam.view_all( rend->mod.bbox );

    rend->manips.push_back( make_shared<visionaray::arcball_manipulator>(rend->cam, mouse::Left) );
    rend->manips.push_back( make_shared<visionaray::pan_manipulator>(rend->cam, mouse::Middle) );
    rend->manips.push_back( make_shared<visionaray::zoom_manipulator>(rend->cam, mouse::Right) );

    glViewport(0, 0, 512, 512);
    rend->host_rt.resize(512, 512);

    glutMainLoop();
}
