// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <thread>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

    #pragma GCC diagnostic ignored "-Wdeprecated"
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#endif

#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#include <GL/glut.h>

#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/gl/bvh_outline_renderer.h>
#include <visionaray/gl/debug_callback.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/area_light.h>
#include <visionaray/bvh.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/host_device_rt.h>
#include <common/make_materials.h>
#include <common/model.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <common/cuda.h>
#endif

#include "call_kernel.h"


using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Switch to use simple but fast plastic material, or a generic material container with
// support for emissive objects
//

#define USE_PLASTIC_MATERIAL 1


//-------------------------------------------------------------------------------------------------
// With emissive materials enabled, assemble light sources from emissive surfaces for
// direct light sampling
//

#define USE_AREA_LIGHT_SAMPLING 0


static_assert(!(USE_PLASTIC_MATERIAL & USE_AREA_LIGHT_SAMPLING), "Invalid configuration");


//-------------------------------------------------------------------------------------------------
// Renderer, stores state, geometry, normals, ...
//

struct renderer : viewer_type
{

//  using scalar_type_cpu           = float;
    using scalar_type_cpu           = simd::float4;
//  using scalar_type_cpu           = simd::float8;
//  using scalar_type_cpu           = simd::float16;
    using scalar_type_gpu           = float;
    using ray_type_cpu              = basic_ray<scalar_type_cpu>;
    using ray_type_gpu              = basic_ray<scalar_type_gpu>;

    using primitive_type            = model::triangle_type;
    using normal_type               = model::normal_type;
    using tex_coord_type            = model::tex_coord_type;
#if USE_PLASTIC_MATERIAL
    using material_type             = plastic<float>;
#else
    using material_type             = generic_material<
                                            emissive<float>,
                                            glass<float>,
                                            matte<float>,
                                            mirror<float>,
                                            plastic<float>
                                            >;
#endif
#if USE_AREA_LIGHT_SAMPLING
    using light_type                = area_light<float, basic_triangle<3, float>>;
#else
    using light_type                = point_light<float>;
#endif

    using host_bvh_type             = index_bvh<primitive_type>;
#ifdef __CUDACC__
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
    using device_tex_type           = cuda_texture<vector<4, unorm<8>>, 2>;
    using device_tex_ref_type       = typename device_tex_type::ref_type;
#endif

    enum bvh_build_strategy
    {
        Binned = 0,  // Binned SAH builder, no spatial splits
        Split        // Split BVH, also binned and with SAH
    };

    enum color_space
    {
        RGB = 0,
        SRGB
    };


    renderer()
        : viewer_type(800, 800, "Visionaray Viewer")
        , host_sched(std::thread::hardware_concurrency())
        , rt(host_device_rt::CPU, true /* direct rendering */)
#ifdef __CUDACC__
        , device_sched(8, 8)
#endif
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

        add_cmdline_option( cl::makeOption<bvh_build_strategy&>({
                { "default",            Binned,         "Binned SAH" },
                { "split",              Split,          "Binned SAH with spatial splits" }
            },
            "bvh",
            cl::Desc("BVH build strategy"),
            cl::ArgRequired,
            cl::init(this->builder)
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

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "bounces",
            cl::Desc("Number of bounces for recursive ray tracing"),
            cl::ArgRequired,
            cl::init(this->bounces)
            ) );

        add_cmdline_option( cl::makeOption<vec3&, cl::ScalarType>(
            [&](StringRef name, StringRef /*arg*/, vec3& value)
            {
                cl::Parser<>()(name + "-r", cmd_line_inst().bump(), value.x);
                cl::Parser<>()(name + "-g", cmd_line_inst().bump(), value.y);
                cl::Parser<>()(name + "-b", cmd_line_inst().bump(), value.z);
            },
            "ambient",
            cl::Desc("Ambient color"),
            cl::ArgDisallowed,
            cl::init(this->ambient)
            ) );

        add_cmdline_option( cl::makeOption<color_space&>({
                { "rgb",                RGB,            "RGB color space for display" },
                { "srgb",               SRGB,           "sRGB color space for display" },
            },
            "colorspace",
            cl::Desc("Color space"),
            cl::ArgRequired,
            cl::init(this->col_space)
            ) );

#ifdef __CUDACC__
        add_cmdline_option( cl::makeOption<host_device_rt::mode&>({
                { "cpu", host_device_rt::CPU, "Rendering on the CPU" },
                { "gpu", host_device_rt::GPU, "Rendering on the GPU" },
            },
            "device",
            cl::Desc("Rendering device"),
            cl::ArgRequired,
            cl::init(rt.current_mode())
            ) );
#endif
    }


    int                                         w               = 800;
    int                                         h               = 800;
    unsigned                                    frame_num       = 0;
    unsigned                                    bounces         = 0;
    unsigned                                    ssaa_samples    = 1;
    algorithm                                   algo            = Simple;
    bvh_build_strategy                          builder         = Binned;
    color_space                                 col_space       = SRGB;
    bool                                        show_hud        = true;
    bool                                        show_hud_ext    = true;
    bool                                        show_bvh        = false;


    std::string                                 filename;
    std::string                                 initial_camera;

    model                                       mod;
    vec3                                        ambient         = vec3(-1.0f);

    host_bvh_type                               host_bvh;
    aligned_vector<material_type>               host_materials;
    aligned_vector<light_type>                  host_lights;
#ifdef __CUDACC__
    device_bvh_type                             device_bvh;
    thrust::device_vector<normal_type>          device_normals;
    thrust::device_vector<tex_coord_type>       device_tex_coords;
    thrust::device_vector<material_type>        device_materials;
    std::map<std::string, device_tex_type>      device_texture_map;
    thrust::device_vector<device_tex_ref_type>  device_textures;
#endif

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
    tbb_sched<ray_type_cpu>                     host_sched;
#else
    tiled_sched<ray_type_cpu>                   host_sched;
#endif
    host_device_rt                              rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>                    device_sched;
#endif
    pinhole_camera                              cam;

    mouse::pos                                  mouse_pos;

    visionaray::frame_counter                   counter;
    gl::bvh_outline_renderer                    outlines;
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

        glColor3f(1.0f, 1.0f, 1.0f);

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

std::istream& operator>>(std::istream& in, pinhole_camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

std::ostream& operator<<(std::ostream& out, pinhole_camera const& cam)
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
        rt.clear_color_buffer();
    }
}


//-------------------------------------------------------------------------------------------------
// HUD
//

void renderer::render_hud()
{
    // gather data to render

    int w = width();
    int h = height();

    int x = visionaray::clamp( mouse_pos.x, 0, w - 1 );
    int y = visionaray::clamp( mouse_pos.y, 0, h - 1 );
    auto color = rt.color();
    auto rgba = color[(h - 1 - y) * w + x];

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


    // render

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

    hud.buffer() << "# Triangles: " << mod.primitives.size();
    hud.print_buffer(300, h * 2 - 34);
    hud.clear_buffer();

    hud.buffer() << "# BVH Nodes/Leaves: " << num_nodes << '/' << num_leaves;
    hud.print_buffer(300, h * 2 - 68);
    hud.clear_buffer();

    hud.buffer() << "SPP: " << std::max(1U, frame_num);
    hud.print_buffer(300, h * 2 - 102);
    hud.clear_buffer();

    hud.buffer() << "Device: " << ( (rt.current_mode() == host_device_rt::GPU) ? "GPU" : "CPU" );
    hud.print_buffer(300, h * 2 - 136);
    hud.clear_buffer();


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
#if !USE_AREA_LIGHT_SAMPLING
    using light_type = point_light<float>;

    light_type light;
    light.set_cl( vec3(1.0, 1.0, 1.0) );
    light.set_kl(1.0);
    light.set_position( cam.eye() );

    host_lights[0] = light;
#endif


    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = this->bounces ? this->bounces : algo == Pathtracing ? 10U : 4U;
    auto epsilon    = std::max( 1E-3f, length(diagonal) * 1E-5f );
    auto amb        = ambient.x >= 0.0f // if set via cmdline
                            ? vec4(ambient, 1.0f)
                            : algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                            ;

    if (rt.current_mode() == host_device_rt::GPU)
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
                amb
                );

        call_kernel( algo, device_sched, kparams, frame_num, ssaa_samples, cam, rt );
#endif
    }
    else if (rt.current_mode() == host_device_rt::CPU)
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
                host_materials.data(),
//              mod.textures.data(),
                host_lights.data(),
                host_lights.data() + host_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                amb
                );

        call_kernel( algo, host_sched, kparams, frame_num, ssaa_samples, cam, rt );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (col_space == SRGB)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }

    rt.display_color_buffer();


    // OpenGL overlay rendering

    if (show_hud)
    {
        render_hud();
    }

    if (show_bvh)
    {
        outlines.frame(cam.get_view_matrix(), cam.get_proj_matrix());
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

    case 'c':
        if (col_space == renderer::RGB)
        {
            col_space = renderer::SRGB;
        }
        else
        {
            col_space = renderer::RGB;
        }
        break;

     case 'h':
        show_hud = !show_hud;
        break;

   case 'm':
#ifdef __CUDACC__
        if (rt.current_mode() == host_device_rt::CPU)
        {
            rt.current_mode() = host_device_rt::GPU;
        }
        else
        {
            rt.current_mode() = host_device_rt::CPU;
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
        if (algo != Pathtracing)
        {
            counter.reset();
            clear_frame();
        }
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
    if (event.buttons() != mouse::NoButton)
    {
        clear_frame();
    }

    mouse_pos = event.pos();
    viewer_type::on_mouse_move(event);
}

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rt.resize(w, h);
    clear_frame();
    viewer_type::on_resize(w, h);
}

int main(int argc, char** argv)
{
    renderer rend;

#ifdef __CUDACC__
    if (rend.rt.direct_rendering() && cuda::init_gl_interop() != cudaSuccess)
    {
        std::cerr << "Cannot initialize CUDA OpenGL interop\n";
        return EXIT_FAILURE;
    }
#endif

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    rend.gl_debug_callback.activate();

    // Load the scene
    std::cout << "Loading model...\n";

    if (!rend.mod.load(rend.filename))
    {
        std::cerr << "Failed loading model\n";
        return EXIT_FAILURE;
    }

    // Convert generic materials to viewer's material type
#if USE_PLASTIC_MATERIAL
    rend.host_materials = make_materials(
            renderer::material_type{},
            rend.mod.materials
            );
#else
    rend.host_materials = make_materials(
            renderer::material_type{},
            rend.mod.materials,
            [](aligned_vector<renderer::material_type>& cont, model::material_type mat)
            {
                // Add emissive material if emissive component > 0
                if (length(mat.ce) > 0.0f)
                {
                    emissive<float> em;
                    em.ce() = from_rgb(mat.ce);
                    em.ls() = 1.0f;
                    cont.emplace_back(em);
                }
                else if (mat.illum == 1)
                {
                    matte<float> ma;
                    ma.ca() = from_rgb(mat.ca);
                    ma.cd() = from_rgb(mat.cd);
                    ma.ka() = 1.0f;
                    ma.kd() = 1.0f;
                    cont.emplace_back(ma);
                }
                else if (mat.illum == 3)
                {
                    mirror<float> mi;
                    mi.cr() = from_rgb(mat.cs);
                    mi.kr() = 1.0f;
                    mi.ior() = spectrum<float>(0.0f);
                    mi.absorption() = spectrum<float>(0.0f);
                    cont.emplace_back(mi);
                }
                else if (mat.illum == 4 && mat.transmission > 0.0f)
                {
                    glass<float> gl;
                    gl.ct() = from_rgb(mat.cd);
                    gl.kt() = 1.0f;
                    gl.cr() = from_rgb(mat.cs);
                    gl.kr() = 1.0f;
                    gl.ior() = from_rgb(mat.ior);
                    cont.push_back(gl);
                }
                else
                {
                    plastic<float> pl;
                    pl.ca() = from_rgb(mat.ca);
                    pl.cd() = from_rgb(mat.cd);
                    pl.cs() = from_rgb(mat.cs);
                    pl.ka() = 1.0f;
                    pl.kd() = 1.0f;
                    pl.ks() = 1.0f;
                    pl.specular_exp() = mat.specular_exp;
                    cont.emplace_back(pl);
                }
            }
            );
#endif

//  timer t;

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend.host_bvh = build<renderer::host_bvh_type>(
            rend.mod.primitives.data(),
            rend.mod.primitives.size(),
            rend.builder == renderer::Split
            );

#if USE_AREA_LIGHT_SAMPLING
    // Loop over all triangles, check if their
    // material is emissive, and if so, build
    // BVHs to create area lights from.

    struct range
    {
        std::size_t begin;
        std::size_t end;
        unsigned    geom_id;
    };

    std::vector<range> ranges;

    for (std::size_t i = 0; i < rend.mod.primitives.size(); ++i)
    {
        auto pi = rend.mod.primitives[i];
        if (rend.host_materials[pi.geom_id].as<emissive<float>>() != nullptr)
        {
            range r;
            r.begin = i;

            std::size_t j = i + 1;
            for (;j < rend.mod.primitives.size(); ++j)
            {
                auto pii = rend.mod.primitives[j];
                if (rend.host_materials[pii.geom_id].as<emissive<float>>() == nullptr
                                || pii.geom_id != pi.geom_id)
                {
                    break;
                }
            }

            r.end = j;
            r.geom_id = pi.geom_id;
            ranges.push_back(r);

            i = r.end - 1;
        }
    }

    for (auto r : ranges)
    {
        for (std::size_t i = r.begin; i != r.end; ++i)
        {
            area_light<float, basic_triangle<3, float>> light(rend.mod.primitives[i]);
            auto mat = *rend.host_materials[r.geom_id].as<emissive<float>>();
            light.set_cl(to_rgb(mat.ce()));
            light.set_kl(mat.ls());
            rend.host_lights.push_back(light);
        }
    }
#else
    rend.host_lights.resize(1);
#endif

    std::cout << "Ready\n";

#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        rend.device_bvh = renderer::device_bvh_type(rend.host_bvh);
        rend.device_normals = rend.mod.geometric_normals;
        rend.device_tex_coords = rend.mod.tex_coords;
        rend.device_materials = rend.host_materials;


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
    catch (std::bad_alloc const&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        rend.device_bvh = renderer::device_bvh_type();
        rend.device_normals.clear();
        rend.device_normals.shrink_to_fit();
        rend.device_tex_coords.clear();
        rend.device_tex_coords.shrink_to_fit();
        rend.device_materials.clear();
        rend.device_materials.shrink_to_fit();
        rend.device_texture_map.clear();
        rend.device_textures.clear();
        rend.device_textures.shrink_to_fit();
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
