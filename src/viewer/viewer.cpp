// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

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

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/texture/texture.h>
#include <visionaray/bvh.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/experimental/tbb_sched.h>
#endif

#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/call_kernel.h>
#include <common/render_bvh.h>
#include <common/timer.h>
#include <common/util.h>

#include "manip/arcball_manipulator.h"
#include "manip/pan_manipulator.h"
#include "manip/zoom_manipulator.h"
#include "default_scenes.h"
#include "model.h"
#include "obj_loader.h"


using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

typedef std::vector<shared_ptr<visionaray::camera_manipulator>> manipulators;


struct renderer
{

//  typedef       float                 scalar_type_cpu;
    typedef simd::float4                scalar_type_cpu;
//  typedef simd::float8                scalar_type_cpu;
    typedef       float                 scalar_type_gpu;
    typedef basic_ray<scalar_type_cpu>  ray_type_cpu;
    typedef basic_ray<scalar_type_gpu>  ray_type_gpu;

    using primitive_type    = model::triangle_list::value_type;
    using normal_type       = model::normal_list::value_type;
    using material_type     = model::mat_list::value_type;

    using host_bvh_type     = bvh<primitive_type>;
#ifdef __CUDACC__
    using device_bvh_type   = device_bvh<primitive_type>;
#endif

    renderer()
        : algo(Simple)
        , w(800)
        , h(800)
        , frame(0)
        , sched_cpu(get_num_processors())
        , down_button(mouse::NoButton)
    {
    }

    algorithm algo;

    int w;
    int h;
    unsigned frame;

    model mod;

    host_bvh_type                           host_bvh;
#ifdef __CUDACC__
    device_bvh_type                         device_bvh;
    thrust::device_vector<normal_type>      device_normals;
    thrust::device_vector<material_type>    device_materials;
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
    tbb_sched<ray_type_cpu>     sched_cpu;
#else
    tiled_sched<ray_type_cpu>   sched_cpu;
#endif
    cpu_buffer_rt               rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>    sched_gpu;
    pixel_unpack_buffer_rt      device_rt;
#endif
    camera                      cam;
    manipulators                manips;

    mouse::button down_button;
    mouse::pos motion_pos;
    mouse::pos down_pos;
    mouse::pos up_pos;

    visionaray::frame_counter   counter;
    bvh_outline_renderer        outlines;
};

renderer* rend = 0;

//auto scene = visionaray::detail::default_generic_prim_scene();


struct configuration
{
    enum device_type
    {
        CPU = 0,
        GPU
    };

    configuration()
        : dev_type(CPU)
        , show_hud(true)
        , show_hud_ext(true)
        , show_bvh(false)
    {
    }

    device_type dev_type;

    bool        show_hud;
    bool        show_hud_ext;
    bool        show_bvh;
};

configuration config;


//-------------------------------------------------------------------------------------------------
// I/O utility for camera lookat only - not fit for the general case!
//

std::istream& operator>>(std::istream& in, camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

std::ostream& operator<<(std::ostream& out, camera const& cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}


void render_hud()
{

    auto w = rend->w;
    auto h = rend->h;

    int x = visionaray::clamp( rend->motion_pos.x, 0, w - 1 );
    int y = visionaray::clamp( rend->motion_pos.y, 0, h - 1 );
    auto color = rend->rt.color();
    auto rgba = color[(h - 1 - y) * w + x];

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w * 2, 0, h * 2);

    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    std::stringstream stream;
    stream << "X: " << x;
    std::string str = stream.str();
    glRasterPos2i(10, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "Y: " << y;
    str = stream.str();
    glRasterPos2i(10, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "W: " << w;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "H: " << h;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream << std::fixed << std::setprecision(2);

    stream.str(std::string());
    stream << "R: " << rgba.x;
    str = stream.str();
    glRasterPos2i(10, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "G: " << rgba.y;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "B: " << rgba.z;
    str = stream.str();
    glRasterPos2i(190, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "FPS: " << rend->counter.register_frame();
    str = stream.str();
    glRasterPos2i(10, h * 2 - 136);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

}

void render_hud_ext()
{

    auto w = rend->w;
    auto h = rend->h;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w * 2, 0, h * 2);

    std::stringstream stream;
    stream << "# Triangles: " << rend->mod.primitives.size();
    std::string str = stream.str();
    glRasterPos2i(300, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

/*    stream.str(std::string());
    stream << "# BVH nodes: " << 1000;
    str = stream.str();
    glRasterPos2i(300, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }*/

    stream.str(std::string());
    stream << "SPP: " << max(1U, rend->frame);
    str = stream.str();
    glRasterPos2i(300, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }


    stream.str(std::string());
    stream << "Device: " << ( (config.dev_type == configuration::GPU) ? "GPU" : "CPU" );
    str = stream.str();
    glRasterPos2i(300, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }


    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

}

void display_func()
{
    using light_type = point_light<float>;

    std::vector<light_type> host_lights;

    light_type light;
    light.set_cl( vec3(1.0, 1.0, 1.0) );
    light.set_kl(1.0);
    light.set_position( rend->cam.eye() );

    host_lights.push_back( light );

    float epsilon = 0.001f;
    vec4 bg_color(0.1, 0.4, 1.0, 1.0);

    if (config.dev_type == configuration::GPU)
    {
#ifdef __CUDACC__
        thrust::device_vector<renderer::device_bvh_type::bvh_ref> device_primitives;

        device_primitives.push_back(rend->device_bvh.ref());

        thrust::device_vector<light_type> device_lights = host_lights;

        auto kparams = make_params<normals_per_face_binding>
        (
            thrust::raw_pointer_cast(device_primitives.data()),
            thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
            thrust::raw_pointer_cast(rend->device_normals.data()),
            thrust::raw_pointer_cast(rend->device_materials.data()),
            thrust::raw_pointer_cast(device_lights.data()),
            thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
            epsilon,
            bg_color
        );

        call_kernel( rend->algo, rend->sched_gpu, kparams, rend->frame, rend->cam, rend->device_rt );
#endif
    }
    else if (config.dev_type == configuration::CPU)
    {
#ifndef __CUDA_ARCH__
        std::vector<renderer::host_bvh_type::bvh_ref> host_primitives;

        host_primitives.push_back(rend->host_bvh.ref());

        auto& mod = rend->mod;

        auto kparams = make_params<normals_per_face_binding>
        (
            host_primitives.data(),
            host_primitives.data() + host_primitives.size(),
            mod.normals.data(),
//            mod.tex_coords.data(),
            mod.materials.data(),
//            mod.textures.data(),
            host_lights.data(),
            host_lights.data() + host_lights.size(),
            epsilon,
            bg_color
        );

        call_kernel( rend->algo, rend->sched_cpu, kparams, rend->frame, rend->cam, rend->rt );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (config.dev_type == configuration::GPU && false /* no direct rendering */)
    {
#ifdef __CUDACC__
//        rend->rt = rend->device_rt;
//        rend->rt.display_color_buffer();
#endif
    }
    else if (config.dev_type == configuration::GPU && true /* direct rendering */)
    {
#ifdef __CUDACC__
        rend->device_rt.display_color_buffer();
#endif
    }
    else
    {
        rend->rt.display_color_buffer();
    }


    // OpenGL overlay rendering

    glColor3f(1.0f, 1.0f, 1.0f);

    if (config.show_hud)
    {
        render_hud();
    }

    if (config.show_hud_ext)
    {
        render_hud_ext();
    }

    if (config.show_bvh)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(rend->cam.get_proj_matrix().data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(rend->cam.get_view_matrix().data());

        rend->outlines.frame();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    glutSwapBuffers();

}

void idle_func()
{
    glutPostRedisplay();
}

void keyboard_func(unsigned char key, int, int)
{
    switch (key)
    {
    case '1':
        std::cout << "Switching algorithm: simple\n";
        rend->algo = Simple;
        rend->counter.reset();
        rend->frame = 0;
        break;

    case '2':
        std::cout << "Switching algorithm: whitted\n";
        rend->algo = Whitted;
        rend->counter.reset();
        rend->frame = 0;
        break;

    case '3':
        std::cout << "Switching algorithm: path tracing\n";
        rend->algo = Pathtracing;
        rend->counter.reset();
        rend->frame = 0;
        break;

    case 'b':
        config.show_bvh = !config.show_bvh;

        if (config.show_bvh)
        {
            rend->outlines.init(rend->host_bvh);
        }

        break;

    case 'm':
#ifdef __CUDACC__
        if (config.dev_type == configuration::CPU)
        {
            config.dev_type = configuration::GPU;
        }
        else
        {
            config.dev_type = configuration::CPU;
        }
        rend->counter.reset();
        rend->frame = 0;
#endif
        break;

    case 'u':
        {
            std::ofstream file("visionaray-camera.txt");
            if (file.good())
            {
                file << rend->cam;
            }
        }
        break;

    case 'v':
        {
            std::ifstream file("visionaray-camera.txt");
            if (file.good())
            {
                file >> rend->cam;
                rend->counter.reset();
                rend->frame = 0;
            }
        }
        break;

    default:
        break;
    }
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

void motion_func(int x, int y)
{
    using namespace visionaray::mouse;

    rend->frame = 0;

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

void reshape_func(int w, int h)
{
    rend->frame = 0;

    glViewport(0, 0, w, h);
    rend->cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->rt.resize(w, h);
#ifdef __CUDACC__
    rend->device_rt.resize(w, h);
#endif
    rend->w = w;
    rend->h = h;
}

void close_func()
{
    delete rend;
}

int main(int argc, char** argv)
{

    if (argc == 1)
    {
        std::cerr << "Usage: viewer FILENAME" << std::endl;
        return EXIT_FAILURE;
    }

    rend = new renderer;

    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(rend->w, rend->h);
    glutCreateWindow("Visionaray GLUT Viewer");
    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutKeyboardFunc(keyboard_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(passive_motion_func);
    glutReshapeFunc(reshape_func);
#ifdef FREEGLUT
    glutCloseFunc(close_func);
#else
    atexit(close_func);
#endif

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "glewInit() failed" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the scene
    std::cout << "Loading model...\n";

    visionaray::load_obj(argv[1], rend->mod);

//  timer t;

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend->host_bvh = build<renderer::host_bvh_type>(rend->mod.primitives.data(), rend->mod.primitives.size());

    std::cout << "Ready\n";

#ifdef __CUDACC__
    // Copy data to GPU
    rend->device_bvh = renderer::device_bvh_type(rend->host_bvh);
    rend->device_normals = rend->mod.normals;
    rend->device_materials = rend->mod.materials;
#endif

//  std::cout << t.elapsed() << std::endl;

    float aspect = rend->w / static_cast<float>(rend->h);

    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->cam.view_all( rend->mod.bbox );

    rend->manips.push_back( make_shared<visionaray::arcball_manipulator>(rend->cam, mouse::Left) );
    rend->manips.push_back( make_shared<visionaray::pan_manipulator>(rend->cam, mouse::Middle) );
    rend->manips.push_back( make_shared<visionaray::zoom_manipulator>(rend->cam, mouse::Right) );

    glutMainLoop();

}
