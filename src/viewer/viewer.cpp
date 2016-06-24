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

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

    #pragma GCC diagnostic ignored "-Wdeprecated"
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#include <GL/gl.h>
#include <GL/glut.h>

#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/render_bvh.h>
#include <visionaray/gl/debug_callback.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/experimental/tbb_sched.h>
#endif

#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/call_kernel.h>
#include <common/model.h>
#include <common/obj_loader.h>
#include <common/timer.h>
#include <common/viewer_glut.h>


using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Renderer, stores state, geometry, normals, ...
//

struct renderer : viewer_type
{

//  using scalar_type_cpu           = float;
    using scalar_type_cpu           = simd::float4;
//  using scalar_type_cpu           = simd::float8;
    using scalar_type_gpu           = float;
    using ray_type_cpu              = basic_ray<scalar_type_cpu>;
    using ray_type_gpu              = basic_ray<scalar_type_gpu>;

    using primitive_type            = model::triangle_list::value_type;
    using normal_type               = model::normal_list::value_type;
    using material_type             = model::mat_list::value_type;

    using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using host_bvh_type             = index_bvh<primitive_type>;
#ifdef __CUDACC__
    using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
#endif

    enum device_type
    {
        CPU = 0,
        GPU
    };


    renderer()
        : viewer_type(800, 800, "Visionaray Viewer")
        , host_sched(std::thread::hardware_concurrency())
        , mouse_pos(0)
    {
        using namespace support;

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "filename",
            cl::Desc("Input file in wavefront obj format"),
            cl::Positional,
            cl::Required,
            cl::init(this->filename)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "camera",
            cl::Desc("Text file with camera parameters"),
            cl::ArgRequired,
            cl::init(this->initial_camera)
            ) );

        add_cmdline_option( cl::makeOption<algorithm&>({
                { "simple",             Simple,         "Simple ray casting kernel" },
                { "whitted",            Whitted,        "Whitted style ray tracing kernel" },
                { "pathtracing",        Pathtracing,    "Pathtracing global illumination kernel" }
            },
            "algorithm",
            cl::Desc("Rendering algorithm"),
            cl::ArgRequired,
            cl::init(this->algo)
            ) );

        add_cmdline_option( cl::makeOption<unsigned&>({
                { "1",      1,      "1x supersampling" },
                { "2",      2,      "2x supersampling" },
                { "4",      4,      "4x supersampling" },
                { "8",      8,      "8x supersampling" }
            },
            "ssaa",
            cl::Desc("Supersampling anti-aliasing factor"),
            cl::ArgRequired,
            cl::init(this->ssaa_samples)
            ) );

#ifdef __CUDACC__
        add_cmdline_option( cl::makeOption<device_type&>({
                { "cpu",                CPU,            "Rendering on the CPU" },
                { "gpu",                GPU,            "Rendering on the GPU" },
            },
            "device",
            cl::Desc("Rendering device"),
            cl::ArgRequired,
            cl::init(this->dev_type)
            ) );
#endif
    }


    int                                     w               = 800;
    int                                     h               = 800;
    unsigned                                frame           = 0;
    unsigned                                ssaa_samples    = 1;
    algorithm                               algo            = Simple;
    device_type                             dev_type        = CPU;
    bool                                    show_hud        = true;
    bool                                    show_hud_ext    = true;
    bool                                    show_bvh        = false;


    std::string                             filename;
    std::string                             initial_camera;

    model                                   mod;

    host_bvh_type                           host_bvh;
#ifdef __CUDACC__
    device_bvh_type                         device_bvh;
    thrust::device_vector<normal_type>      device_normals;
    thrust::device_vector<material_type>    device_materials;
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
    tbb_sched<ray_type_cpu>                 host_sched;
#else
    tiled_sched<ray_type_cpu>               host_sched;
#endif
    host_render_target_type                 host_rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>                device_sched;
    device_render_target_type               device_rt;
#endif
    camera                                  cam;

    mouse::pos                              mouse_pos;

    visionaray::frame_counter               counter;
    detail::bvh_outline_renderer            outlines;
    gl::debug_callback                      gl_debug_callback;

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

private:

    void render_hud();
    void render_hud_ext();

};


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


//-------------------------------------------------------------------------------------------------
// HUD
//

void renderer::render_hud()
{

    auto w = width();
    auto h = height();

    int x = visionaray::clamp( mouse_pos.x, 0, w - 1 );
    int y = visionaray::clamp( mouse_pos.y, 0, h - 1 );
    auto color = host_rt.color();
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
    stream << "FPS: " << counter.register_frame();
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

void renderer::render_hud_ext()
{

    auto w = width();
    auto h = height();

    int num_nodes = 0;
    int num_leaves = 0;

    traverse_depth_first(
        host_bvh,
        [&](renderer::host_bvh_type::node_type const& node)
        {
            ++num_nodes;

            if (is_leaf(node))
            {
                ++num_leaves;
            }
        }
        );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w * 2, 0, h * 2);

    std::stringstream stream;
    stream << "# Triangles: " << mod.primitives.size();
    std::string str = stream.str();
    glRasterPos2i(300, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "# BVH Nodes/Leaves: " << num_nodes << '/' << num_leaves;
    str = stream.str();
    glRasterPos2i(300, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "SPP: " << max(1U, frame);
    str = stream.str();
    glRasterPos2i(300, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }


    stream.str(std::string());
    stream << "Device: " << ( (dev_type == renderer::GPU) ? "GPU" : "CPU" );
    str = stream.str();
    glRasterPos2i(300, h * 2 - 136);
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

void renderer::on_close()
{
    outlines.destroy();
}

void renderer::on_display()
{
    using light_type = point_light<float>;

    aligned_vector<light_type> host_lights;

    light_type light;
    light.set_cl( vec3(1.0, 1.0, 1.0) );
    light.set_kl(1.0);
    light.set_position( cam.eye() );

    host_lights.push_back( light );

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = algo == Pathtracing ? 10U : 4U;
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );

    vec4 bg_color(0.1, 0.4, 1.0, 1.0);

    if (dev_type == renderer::GPU)
    {
#ifdef __CUDACC__
        thrust::device_vector<renderer::device_bvh_type::bvh_ref> device_primitives;

        device_primitives.push_back(device_bvh.ref());

        thrust::device_vector<light_type> device_lights = host_lights;

        auto kparams = make_kernel_params(
                normals_per_face_binding{},
                thrust::raw_pointer_cast(device_primitives.data()),
                thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
                thrust::raw_pointer_cast(device_normals.data()),
                thrust::raw_pointer_cast(device_materials.data()),
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                );

        call_kernel( algo, device_sched, kparams, frame, ssaa_samples, cam, device_rt );
#endif
    }
    else if (dev_type == renderer::CPU)
    {
#ifndef __CUDA_ARCH__
        aligned_vector<renderer::host_bvh_type::bvh_ref> host_primitives;

        host_primitives.push_back(host_bvh.ref());

        auto kparams = make_kernel_params(
                normals_per_face_binding{},
                host_primitives.data(),
                host_primitives.data() + host_primitives.size(),
                mod.geometric_normals.data(),
//              mod.tex_coords.data(),
                mod.materials.data(),
//              mod.textures.data(),
                host_lights.data(),
                host_lights.data() + host_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                );

        call_kernel( algo, host_sched, kparams, frame, ssaa_samples, cam, host_rt );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_FRAMEBUFFER_SRGB);

    if (dev_type == renderer::GPU && false /* no direct rendering */)
    {
#ifdef __CUDACC__
//        host_rt = device_rt;
//        host_rt.display_color_buffer();
#endif
    }
    else if (dev_type == renderer::GPU && true /* direct rendering */)
    {
#ifdef __CUDACC__
        device_rt.display_color_buffer();
#endif
    }
    else
    {
        host_rt.display_color_buffer();
    }


    // OpenGL overlay rendering

    glColor3f(1.0f, 1.0f, 1.0f);

    if (show_hud)
    {
        render_hud();
    }

    if (show_hud_ext)
    {
        render_hud_ext();
    }

    if (show_bvh)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(cam.get_proj_matrix().data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(cam.get_view_matrix().data());

        outlines.frame();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }
}

void renderer::on_key_press(key_event const& event)
{
    static const std::string camera_filename = "visionaray-camera.txt";

    switch (event.key())
    {
    case '1':
        std::cout << "Switching algorithm: simple\n";
        algo = Simple;
        counter.reset();
        frame = 0;
        break;

    case '2':
        std::cout << "Switching algorithm: whitted\n";
        algo = Whitted;
        counter.reset();
        frame = 0;
        break;

    case '3':
        std::cout << "Switching algorithm: path tracing\n";
        algo = Pathtracing;
        counter.reset();
        frame = 0;
        break;

    case 'b':
        show_bvh = !show_bvh;

        if (show_bvh)
        {
            outlines.init(host_bvh);
        }

        break;

    case 'm':
#ifdef __CUDACC__
        if (dev_type == renderer::CPU)
        {
            dev_type = renderer::GPU;
        }
        else
        {
            dev_type = renderer::CPU;
        }
        counter.reset();
        frame = 0;
#endif
        break;

    case 's':
        ssaa_samples *= 2;
        if (ssaa_samples > 8)
        {
            ssaa_samples = 1;
        }
        std::cout << "Use " << ssaa_samples << "x supersampling anti-aliasing\n";
        counter.reset();
        break;

    case 'u':
        {
            std::ofstream file( camera_filename );
            if (file.good())
            {
                std::cout << "Storing camera to file: " << camera_filename << '\n';
                file << cam;
            }
        }
        break;

    case 'v':
        {
            std::ifstream file( camera_filename );
            if (file.good())
            {
                file >> cam;
                counter.reset();
                frame = 0;
                std::cout << "Load camera from file: " << camera_filename << '\n';
            }
        }
        break;

    default:
        break;
    }

    viewer_type::on_key_press(event);
}

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.get_buttons() != mouse::NoButton)
    {
        frame = 0;
    }

    mouse_pos = event.get_pos();
    viewer_type::on_mouse_move(event);
}

void renderer::on_resize(int w, int h)
{
    frame = 0;

    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);
#ifdef __CUDACC__
    device_rt.resize(w, h);
#endif
    viewer_type::on_resize(w, h);
}

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

    rend.gl_debug_callback.activate();

    // Load the scene
    std::cout << "Loading model...\n";

    try
    {
        visionaray::load_obj(rend.filename, rend.mod);
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

//  timer t;

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend.host_bvh = build<renderer::host_bvh_type>(rend.mod.primitives.data(), rend.mod.primitives.size());

    std::cout << "Ready\n";

#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        rend.device_bvh = renderer::device_bvh_type(rend.host_bvh);
        rend.device_normals = rend.mod.geometric_normals;
        rend.device_materials = rend.mod.materials;
    }
    catch (std::bad_alloc&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        rend.device_bvh = renderer::device_bvh_type();
        rend.device_normals.resize(0);
        rend.device_materials.resize(0);
    }
#endif

//  std::cout << t.elapsed() << std::endl;

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    // Load camera from file or set view-all
    std::ifstream file(rend.initial_camera);
    if (file.good())
    {
        file >> rend.cam;
    }
    else
    {
        rend.cam.view_all( rend.mod.bbox );
    }

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();

}
