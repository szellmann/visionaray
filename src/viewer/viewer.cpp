// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <map>
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

    using primitive_type            = model::triangle_type;
    using normal_type               = model::normal_type;
    using tex_coord_type            = model::tex_coord_type;
    using material_type             = model::material_type;

    using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using host_bvh_type             = index_bvh<primitive_type>;
#ifdef __CUDACC__
    using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
    using device_tex_type           = cuda_texture<vector<4, unorm<8>>, 2>;
    using device_tex_ref_type       = typename device_tex_type::ref_type;
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


    int                                         w               = 800;
    int                                         h               = 800;
    unsigned                                    frame_num       = 0;
    unsigned                                    ssaa_samples    = 1;
    algorithm                                   algo            = Simple;
    device_type                                 dev_type        = CPU;
    bool                                        show_hud        = true;
    bool                                        show_hud_ext    = true;
    bool                                        show_bvh        = false;


    std::string                                 filename;
    std::string                                 initial_camera;

    model                                       mod;

    host_bvh_type                               host_bvh;
#ifdef __CUDACC__
    device_bvh_type                             device_bvh;
    thrust::device_vector<normal_type>          device_normals;
    thrust::device_vector<tex_coord_type>       device_tex_coords;
    thrust::device_vector<material_type>        device_materials;
    std::map<std::string, device_tex_type>      device_texture_map;
    thrust::device_vector<device_tex_ref_type>  device_textures;
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
    tbb_sched<ray_type_cpu>                     host_sched;
#else
    tiled_sched<ray_type_cpu>                   host_sched;
#endif
    host_render_target_type                     host_rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>                    device_sched;
    device_render_target_type                   device_rt;
#endif
    camera                                      cam;

    mouse::pos                                  mouse_pos;

    visionaray::frame_counter                   counter;
    detail::bvh_outline_renderer                outlines;
    gl::debug_callback                          gl_debug_callback;

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

private:

    void clear_frame();
    void render_hud();
    void render_hud_ext();

};


//-------------------------------------------------------------------------------------------------
// Utility for HUD rendering
//

class hud_util
{
public:

    // Stream to buffer string that will be printed
    std::stringstream& buffer()
    {
        return buffer_;
    }

    // Clear string buffer
    void clear_buffer()
    {
        buffer_.str(std::string());
    }

    // Print buffer with origin at pixel (x,y)
    void print_buffer(int x, int y)
    {
        glRasterPos2i(x, y);

        std::string str = buffer_.str();

        for (size_t i = 0; i < str.length(); ++i)
        {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
        }
    }

private:

    std::stringstream buffer_;

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
// If path tracing, clear frame buffer and reset frame counter
//

void renderer::clear_frame()
{
    frame_num = 0;

    if (algo == Pathtracing)
    {
        host_rt.clear_color_buffer();
#ifdef __CUDACC__
        device_rt.clear_color_buffer();
#endif
    }
}


//-------------------------------------------------------------------------------------------------
// HUD
//

void renderer::render_hud()
{
    int w = width();
    int h = height();

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


    hud_util hud;

    hud.buffer() << "X: " << x;
    hud.print_buffer(10, h * 2 - 34);
    hud.clear_buffer();

    hud.buffer() << "Y: " << y;
    hud.print_buffer(10, h * 2 - 68);
    hud.clear_buffer();

    hud.buffer() << "W: " << w;
    hud.print_buffer(100, h * 2 - 34);
    hud.clear_buffer();

    hud.buffer() << "H: " << h;
    hud.print_buffer(100, h * 2 - 68);
    hud.clear_buffer();

    hud.buffer() << std::fixed << std::setprecision(2);

    hud.buffer() << "R: " << rgba.x;
    hud.print_buffer(10, h * 2 - 102);
    hud.clear_buffer();

    hud.buffer() << "G: " << rgba.y;
    hud.print_buffer(100, h * 2 - 102);
    hud.clear_buffer();

    hud.buffer() << "B: " << rgba.z;
    hud.print_buffer(190, h * 2 - 102);
    hud.clear_buffer();

    hud.buffer() << "FPS: " << counter.register_frame();
    hud.print_buffer(10, h * 2 - 136);
    hud.clear_buffer();


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


    hud_util hud;

    hud.buffer() << "# Triangles: " << mod.primitives.size();
    hud.print_buffer(300, h * 2 - 34);
    hud.clear_buffer();

    hud.buffer() << "# BVH Nodes/Leaves: " << num_nodes << '/' << num_leaves;
    hud.print_buffer(300, h * 2 - 68);
    hud.clear_buffer();

    hud.buffer() << "SPP: " << max(1U, frame_num);
    hud.print_buffer(300, h * 2 - 102);
    hud.clear_buffer();

    hud.buffer() << "Device: " << ( (dev_type == renderer::GPU) ? "GPU" : "CPU" );
    hud.print_buffer(300, h * 2 - 136);
    hud.clear_buffer();


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
//              thrust::raw_pointer_cast(device_tex_coords.data()),
                thrust::raw_pointer_cast(device_materials.data()),
//              thrust::raw_pointer_cast(device_textures.data()),
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                );

        call_kernel( algo, device_sched, kparams, frame_num, ssaa_samples, cam, device_rt );
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

        call_kernel( algo, host_sched, kparams, frame_num, ssaa_samples, cam, host_rt );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_FRAMEBUFFER_SRGB);

    if (dev_type == renderer::GPU && false /* no direct rendering */)
    {
#ifdef __CUDACC__
//        device_rt.display_color_buffer();
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
        clear_frame();
        break;

    case '2':
        std::cout << "Switching algorithm: whitted\n";
        algo = Whitted;
        counter.reset();
        clear_frame();
        break;

    case '3':
        std::cout << "Switching algorithm: path tracing\n";
        algo = Pathtracing;
        counter.reset();
        clear_frame();
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
        clear_frame();
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
        clear_frame();
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
                clear_frame();
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
        clear_frame();
    }

    mouse_pos = event.get_pos();
    viewer_type::on_mouse_move(event);
}

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);
#ifdef __CUDACC__
    device_rt.resize(w, h);
#endif
    clear_frame();
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
        rend.device_tex_coords = rend.mod.tex_coords;
        rend.device_materials = rend.mod.materials;


        // Copy textures and texture references to the GPU

        rend.device_textures.resize(rend.mod.textures.size());

        for (auto const& pair_host_tex : rend.mod.texture_map)
        {
            auto const& host_tex = pair_host_tex.second;
            renderer::device_tex_type device_tex(pair_host_tex.second);
            auto const& p = rend.device_texture_map.emplace(pair_host_tex.first, std::move(device_tex));

            assert(p.second /* inserted */);

            auto it = p.first;

            // Texture references ensure that we don't allocate storage
            // for the same texture map more than once.
            // By checking if the pointer in the ref contains the
            // address of the first texel of the map, we can identify
            // which texture_ref references which texture and recreate
            // that relation on the GPU.
            for (size_t i = 0; i < rend.mod.textures.size(); ++i)
            {
                if (rend.mod.textures[i].data() == host_tex.data())
                {
                    rend.device_textures[i] = renderer::device_tex_ref_type(it->second);
                }
            }
        }

        // Place some dummy textures where geometry has no texture
        for (size_t i = 0; i < rend.mod.textures.size(); ++i)
        {
            if (rend.mod.textures[i].width() == 0 || rend.mod.textures[i].height() == 0)
            {
                vector<4, unorm<8>>* dummy = nullptr;
                renderer::device_tex_type device_tex(dummy, 0, 0, Clamp, Nearest);

                // Try to insert the dummy texture into the
                // device texture map...
                auto p = rend.device_texture_map.emplace("", std::move(device_tex));

                // ... but maybe a dummy texture was already
                // inserted, then just find that
                if (!p.second)
                {
                    auto it = rend.device_texture_map.find("");
                    rend.device_textures[i] = renderer::device_tex_ref_type(it->second);

                }
                else
                {
                    auto it = p.first;
                    rend.device_textures[i] = renderer::device_tex_ref_type(it->second);
                }
            }
        }
    }
    catch (std::bad_alloc&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        rend.device_bvh = renderer::device_bvh_type();
        rend.device_normals.resize(0);
        rend.device_tex_coords.resize(0);
        rend.device_materials.resize(0);
        rend.device_textures.resize(0);
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
