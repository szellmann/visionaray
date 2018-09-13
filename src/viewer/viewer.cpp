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

#include <imgui.h>

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
#include <common/make_materials.h>
#include <common/model.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <common/cuda.h>
#endif

#include "call_kernel.h"
#include "host_device_rt.h"
#include "render.h"


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
//  using scalar_type_cpu           = simd::float16;
    using scalar_type_gpu           = float;
    using ray_type_cpu              = basic_ray<scalar_type_cpu>;
    using ray_type_gpu              = basic_ray<scalar_type_gpu>;

    using primitive_type            = model::triangle_type;
    using normal_type               = model::normal_type;
    using tex_coord_type            = model::tex_coord_type;
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


    renderer()
        : viewer_type(800, 800, "Visionaray Viewer")
        , host_sched(std::thread::hardware_concurrency())
        , rt(host_device_rt::CPU, true /* direct rendering */, host_device_rt::SRGB)
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

        add_cmdline_option( cl::makeOption<host_device_rt::color_space_type&>({
                { "rgb",  host_device_rt::RGB,  "RGB color space for display" },
                { "srgb", host_device_rt::SRGB, "sRGB color space for display" },
            },
            "colorspace",
            cl::Desc("Color space"),
            cl::ArgRequired,
            cl::init(rt.color_space())
            ) );

#ifdef __CUDACC__
        add_cmdline_option( cl::makeOption<host_device_rt::mode_type&>({
                { "cpu", host_device_rt::CPU, "Rendering on the CPU" },
                { "gpu", host_device_rt::GPU, "Rendering on the GPU" },
            },
            "device",
            cl::Desc("Rendering device"),
            cl::ArgRequired,
            cl::init(rt.mode())
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
    bool                                        show_hud        = true;
    bool                                        show_hud_ext    = true;
    bool                                        show_bvh        = false;


    std::string                                 filename;
    std::string                                 initial_camera;

    model                                       mod;
    vec3                                        ambient         = vec3(-1.0f);

    host_bvh_type                               host_bvh;
    aligned_vector<plastic<float>>              plastic_materials;
    aligned_vector<generic_material_t>          generic_materials;
    aligned_vector<point_light<float>>          point_lights;
    aligned_vector<area_light<float,
                   basic_triangle<3, float>>>   area_lights;
#ifdef __CUDACC__
    device_bvh_type                             device_bvh;
    thrust::device_vector<normal_type>          device_geometric_normals;
    thrust::device_vector<normal_type>          device_shading_normals;
    thrust::device_vector<tex_coord_type>       device_tex_coords;
    thrust::device_vector<plastic<float>>       device_plastic_materials;
    thrust::device_vector<generic_material_t>   device_generic_materials;
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


    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Settings", &show_hud);

    ImGui::Text("X: %4d", x);
    ImGui::SameLine();
    ImGui::Text("W: %4d", w);
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::Text("# Triangles: %zu", mod.primitives.size());

    ImGui::Text("Y: %4d", y);
    ImGui::SameLine();
    ImGui::Text("H: %4d", h);
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::Text("# BVH Nodes/Leaves: %d/%d", num_nodes, num_leaves);

    ImGui::Text("R: %5.2f", rgba.x);
    ImGui::SameLine();
    ImGui::Text("G: %5.2f", rgba.y);
    ImGui::SameLine();
    ImGui::Text("B: %5.2f", rgba.z);

    ImGui::Text("FPS: %5.2f", counter.register_frame());
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::Text("SPP: %6u", std::max(1U, frame_num));
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::Text("Device: %s", rt.mode() == host_device_rt::GPU ? "GPU" : "CPU");

    ImGui::End();
}

void renderer::on_close()
{
    outlines.destroy();
}

void renderer::on_display()
{
    point_light<float>& headlight = point_lights[0];
    headlight.set_cl( vec3(1.0, 1.0, 1.0) );
    headlight.set_kl(1.0);
    headlight.set_position( cam.eye() );
    headlight.set_constant_attenuation(1.0);
    headlight.set_linear_attenuation(0.0);
    headlight.set_quadratic_attenuation(0.0);

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = this->bounces ? this->bounces : algo == Pathtracing ? 10U : 4U;
    auto epsilon    = std::max( 1E-3f, length(diagonal) * 1E-5f );
    auto amb        = ambient.x >= 0.0f // if set via cmdline
                            ? vec4(ambient, 1.0f)
                            : vec4(0.0)
                            ;

    if (rt.mode() == host_device_rt::CPU)
    {
        if (area_lights.size() > 0 && algo == Pathtracing)
        {
            render_generic_material_cpp(
                    host_bvh,
                    mod.geometric_normals,
                    mod.shading_normals,
                    mod.tex_coords,
                    generic_materials,
                    mod.textures,
                    area_lights,
                    bounces,
                    epsilon,
                    vec4(background_color(), 1.0f),
                    amb,
                    rt,
                    host_sched,
                    cam,
                    frame_num,
                    algo,
                    ssaa_samples
                    );
        }
        else
        {
            render_plastic_cpp(
                    host_bvh,
                    mod.geometric_normals,
                    mod.shading_normals,
                    mod.tex_coords,
                    plastic_materials,
                    mod.textures,
                    point_lights,
                    bounces,
                    epsilon,
                    vec4(background_color(), 1.0f),
                    amb,
                    rt,
                    host_sched,
                    cam,
                    frame_num,
                    algo,
                    ssaa_samples
                    );
        }
    }
#ifdef __CUDACC__
    else if (rt.mode() == host_device_rt::GPU)
    {
        if (area_lights.size() > 0 && algo == Pathtracing)
        {
            render_generic_material_cu(
                    device_bvh,
                    device_geometric_normals,
                    device_shading_normals,
                    device_tex_coords,
                    device_generic_materials,
                    device_textures,
                    area_lights,
                    bounces,
                    epsilon,
                    vec4(background_color(), 1.0f),
                    amb,
                    rt,
                    device_sched,
                    cam,
                    frame_num,
                    algo,
                    ssaa_samples
                    );
        }
        else
        {
            render_plastic_cu(
                    device_bvh,
                    device_geometric_normals,
                    device_shading_normals,
                    device_tex_coords,
                    device_plastic_materials,
                    device_textures,
                    point_lights,
                    bounces,
                    epsilon,
                    vec4(background_color(), 1.0f),
                    amb,
                    rt,
                    device_sched,
                    cam,
                    frame_num,
                    algo,
                    ssaa_samples
                    );
        }
    }
#endif

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
        if (rt.color_space() == host_device_rt::RGB)
        {
            rt.color_space() = host_device_rt::SRGB;
        }
        else
        {
            rt.color_space() = host_device_rt::RGB;
        }
        break;

     case 'h':
        show_hud = !show_hud;
        break;

   case 'm':
#ifdef __CUDACC__
        if (rt.mode() == host_device_rt::CPU)
        {
            rt.mode() = host_device_rt::GPU;
        }
        else
        {
            rt.mode() = host_device_rt::CPU;
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

//  timer t;

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend.host_bvh = build<renderer::host_bvh_type>(
            rend.mod.primitives.data(),
            rend.mod.primitives.size(),
            rend.builder == renderer::Split
            );


    // Generate a list with plastic materials
    rend.plastic_materials = make_materials(
            plastic<float>{},
            rend.mod.materials
            );

    // Generate another list with generic materials
    rend.generic_materials = make_materials(
            generic_material_t{},
            rend.mod.materials,
            [](aligned_vector<generic_material_t>& cont, model::material_type mat)
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
        if (rend.generic_materials[pi.geom_id].as<emissive<float>>() != nullptr)
        {
            range r;
            r.begin = i;

            std::size_t j = i + 1;
            for (;j < rend.mod.primitives.size(); ++j)
            {
                auto pii = rend.mod.primitives[j];
                if (rend.generic_materials[pii.geom_id].as<emissive<float>>() == nullptr
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

    // Build vector with area light sources
    for (auto r : ranges)
    {
        for (std::size_t i = r.begin; i != r.end; ++i)
        {
            area_light<float, basic_triangle<3, float>> light(rend.mod.primitives[i]);
            auto mat = *rend.generic_materials[r.geom_id].as<emissive<float>>();
            light.set_cl(to_rgb(mat.ce()));
            light.set_kl(mat.ls());
            rend.area_lights.push_back(light);
        }
    }

    // Make room for one (head) light
    rend.point_lights.resize(1);

    std::cout << "Ready\n";

#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        rend.device_bvh = renderer::device_bvh_type(rend.host_bvh);
        rend.device_geometric_normals = rend.mod.geometric_normals;
        rend.device_shading_normals = rend.mod.shading_normals;
        rend.device_tex_coords = rend.mod.tex_coords;
        rend.device_plastic_materials = rend.plastic_materials;
        rend.device_generic_materials = rend.generic_materials;


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
        rend.device_geometric_normals.clear();
        rend.device_geometric_normals.shrink_to_fit();
        rend.device_shading_normals.clear();
        rend.device_shading_normals.shrink_to_fit();
        rend.device_tex_coords.clear();
        rend.device_tex_coords.shrink_to_fit();
        rend.device_plastic_materials.clear();
        rend.device_plastic_materials.shrink_to_fit();
        rend.device_generic_materials.clear();
        rend.device_generic_materials.shrink_to_fit();
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
