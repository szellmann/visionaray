// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>

#if VSNRAY_COMMON_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

#include <imgui.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/area_light.h>
#include <visionaray/bvh.h>
#include <visionaray/environment_light.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>
#include <visionaray/spot_light.h>
#include <visionaray/thin_lens_camera.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#include <common/input/keyboard.h>
#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/bvh_outline_renderer.h>
#include <common/gl_debug_callback.h>
#include <common/inifile.h>
#include <common/make_materials.h>
#include <common/make_texture.h>
#include <common/model.h>
#include <common/image.h>
#include <common/sg.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#if VSNRAY_COMMON_HAVE_PTEX
#include <common/ptex.h>
#endif

#include "call_kernel.h"
#include "host_device_rt.h"
#include "render.h"


using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Helpers
//

enum class copy_kind
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
};

template <typename DestInstances, typename DestTopLevel, typename SourceInstances, typename SourceTopLevel>
static void copy_bvhs(
        DestInstances&         dest_instance_bvhs,
        DestTopLevel&          dest_top_level_bvh,
        SourceInstances const& source_instance_bvhs,
        SourceTopLevel const&  source_top_level_bvh,
        copy_kind              ck
        )
{
    // Build up lower level bvhs first
    dest_instance_bvhs.resize(source_instance_bvhs.size());
    for (size_t i = 0; i < source_instance_bvhs.size(); ++i)
    {
        dest_instance_bvhs[i] = typename DestInstances::value_type(source_instance_bvhs[i]);
    }

    // Make a deep copy of the top level bvh
    if (source_top_level_bvh.num_primitives() > 0)
    {
        dest_top_level_bvh.primitives().resize(source_top_level_bvh.num_primitives());
        dest_top_level_bvh.nodes().resize(source_top_level_bvh.num_nodes());
        dest_top_level_bvh.indices().resize(source_top_level_bvh.num_indices());

        // Linear search for each inst: This may obviously become fairly inefficient..
        for (size_t i = 0; i < source_top_level_bvh.num_primitives(); ++i)
        {
            size_t index = size_t(-1);

            for (size_t j = 0; j < source_instance_bvhs.size(); ++j)
            {
                if (source_instance_bvhs[j].ref() == source_top_level_bvh.primitive(i).get_ref())
                {
                    index = j;
                    break;
                }
            }

            assert(index < size_t(-1));

            int indirect_index = source_top_level_bvh.indices()[i];

            dest_top_level_bvh.primitives()[indirect_index] = {
                    dest_instance_bvhs[index].ref(),
                    mat4x3(
                        inverse(source_top_level_bvh.primitive(i).affine_inv()),
                        -source_top_level_bvh.primitive(i).trans_inv()
                        )
                    };
        }

        // Copy nodes and indices
        if (ck == copy_kind::HostToHost)
        {
        }
#if VSNRAY_COMMON_HAVE_CUDA
        else if (ck == copy_kind::HostToDevice)
        {
            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.nodes()),
                    detail::get_pointer(source_top_level_bvh.nodes()),
                    sizeof(bvh_node) * source_top_level_bvh.num_nodes(),
                    cudaMemcpyHostToDevice
                    );

            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.indices()),
                    detail::get_pointer(source_top_level_bvh.indices()),
                    sizeof(unsigned) * source_top_level_bvh.num_indices(),
                    cudaMemcpyHostToDevice
                    );
        }
        else if (ck == copy_kind::DeviceToHost)
        {
            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.nodes()),
                    detail::get_pointer(source_top_level_bvh.nodes()),
                    sizeof(bvh_node) * source_top_level_bvh.num_nodes(),
                    cudaMemcpyDeviceToHost
                    );

            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.indices()),
                    detail::get_pointer(source_top_level_bvh.indices()),
                    sizeof(unsigned) * source_top_level_bvh.num_indices(),
                    cudaMemcpyDeviceToHost
                    );
        }
        else if (ck == copy_kind::DeviceToDevice)
        {
            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.nodes()),
                    detail::get_pointer(source_top_level_bvh.nodes()),
                    sizeof(bvh_node) * source_top_level_bvh.num_nodes(),
                    cudaMemcpyDeviceToDevice
                    );

            cudaMemcpy(
                    (void*)detail::get_pointer(dest_top_level_bvh.indices()),
                    detail::get_pointer(source_top_level_bvh.indices()),
                    sizeof(unsigned) * source_top_level_bvh.num_indices(),
                    cudaMemcpyDeviceToDevice
                    );
        }
#else
        else
        {
            assert(0);
        }
#endif
    }
}


//-------------------------------------------------------------------------------------------------
// Renderer, stores state, geometry, normals, ...
//

struct renderer : viewer_type
{
    using primitive_type            = model::triangle_type;
    using normal_type               = model::normal_type;
    using tex_coord_type            = model::tex_coord_type;
    using color_type                = model::color_type;
    using host_bvh_type             = index_bvh<primitive_type>;
#if VSNRAY_COMMON_HAVE_CUDA
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
    using device_tex_type           = cuda_texture<vector<4, unorm<8>>, 2>;
    using device_tex_ref_type       = typename device_tex_type::ref_type;
#endif

    enum bvh_build_strategy
    {
        Binned = 0, // Binned SAH builder, no spatial splits
        Split,      // Split BVH, also binned and with SAH
        LBVH,       // LBVH builder on the CPU
    };

    enum texture_format { Ptex, UV };

    renderer()
        : viewer_type(800, 800, "Visionaray Viewer")
        , host_sched(std::thread::hardware_concurrency())
        , rt(
            host_device_rt::CPU,
            true /* double buffering */,
            true /* direct rendering */,
            host_device_rt::SRGB
            )
#if VSNRAY_COMMON_HAVE_CUDA
        , device_sched(8, 8)
#endif
        , env_map(0, 0)
        , mouse_pos(0)
    {
        using namespace support;

        // Init null environment light
        env_light.texture() = texture_ref<vec4, 2>(env_map);

        // Parse inifile (but cmdline overrides!)
        parse_inifile({ "vsnray-viewer.ini", "viewer.ini" });

        // Add cmdline options
        add_cmdline_option( cl::makeOption<std::set<std::string>&>(
            cl::Parser<>(),
            "filenames",
            cl::Desc("Input files in wavefront obj format"),
            cl::Positional,
            cl::OneOrMore,
            cl::init(filenames)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "camera",
            cl::Desc("Text file with camera parameters"),
            cl::ArgRequired,
            cl::init(this->initial_camera)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "screenshotbasename",
            cl::Desc("Base name (w/o suffix!) for screenshot files"),
            cl::ArgRequired,
            cl::init(this->screenshot_file_base)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "envmap",
            cl::Desc("HDR environment map"),
            cl::ArgRequired,
            cl::init(this->env_map_filename)
            ) );

        add_cmdline_option( cl::makeOption<algorithm&>({
                { "simple",             Simple,         "Simple ray casting kernel" },
                { "whitted",            Whitted,        "Whitted style ray tracing kernel" },
                { "pathtracing",        Pathtracing,    "Pathtracing global illumination kernel" },
                { "costs",              Costs,          "BVH cost kernel" }
            },
            "algorithm",
            cl::Desc("Rendering algorithm"),
            cl::ArgRequired,
            cl::init(this->algo)
            ) );

        add_cmdline_option( cl::makeOption<bvh_build_strategy&>({
                { "default",            Binned,         "Binned SAH" },
                { "split",              Split,          "Binned SAH with spatial splits" },
                { "lbvh",               LBVH,           "LBVH (CPU)" }
            },
            "bvh",
            cl::Desc("BVH build strategy"),
            cl::ArgRequired,
            cl::init(this->build_strategy)
            ) );

        // The following two options both manipulate spp
        add_cmdline_option( cl::makeOption<unsigned&>({
                { "1",      1,      "1x supersampling" },
                { "2",      2,      "2x supersampling" },
                { "4",      4,      "4x supersampling" },
                { "8",      8,      "8x supersampling" }
            },
            "ssaa",
            cl::Desc("Supersampling anti-aliasing factor"),
            cl::ArgRequired,
            cl::init(this->spp)
            ) );

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "spp",
            cl::Desc("Pixels per sample for path tracing"),
            cl::ArgRequired,
            cl::init(this->spp)
            ) );

        add_cmdline_option( cl::makeOption<bool&>(
            cl::Parser<>(),
            "headlight",
            cl::Desc("Activate headlight"),
            cl::ArgRequired,
            cl::init(this->use_headlight)
            ) );

        add_cmdline_option( cl::makeOption<bool&>(
            cl::Parser<>(),
            "groundplane",
            cl::Desc("Add a ground plane"),
            cl::ArgRequired,
            cl::init(this->use_groundplane)
            ) );

        add_cmdline_option( cl::makeOption<bool&>(
            cl::Parser<>(),
            "dof",
            cl::Desc("Activate depth of field"),
            cl::ArgRequired,
            cl::init(this->use_dof)
            ) );

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "bounces",
            cl::Desc("Number of bounces for recursive ray tracing"),
            cl::ArgRequired,
            cl::init(this->bounces)
            ) );

        add_cmdline_option( cl::makeOption<unsigned&>(
            cl::Parser<>(),
            "frames",
            cl::Desc("Number of path tracer convergence frames"),
            cl::ArgRequired,
            cl::init(this->frames)
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

#if VSNRAY_COMMON_HAVE_CUDA
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

    void parse_inifile(std::set<std::string> filenames)
    {
        // First parse base's options
        viewer_base::parse_inifile(filenames);

        // Process the first (if any) valid inifile
        for (auto filename : filenames)
        {
            inifile ini(filename);

            if (ini.good())
            {
                inifile::error_code err = inifile::Ok;

                // algorithm
                std::string algo = "";
                err = ini.get_string("algorithm", algo);
                if (err == inifile::Ok)
                {
                    if (algo == "simple")
                    {
                        this->algo = Simple;
                    }
                    else if (algo == "whitted")
                    {
                        this->algo = Whitted;
                    }
                    else if (algo == "pathtracing")
                    {
                        this->algo = Pathtracing;
                    }
                    else if (algo == "costs")
                    {
                        this->algo = Costs;
                    }
                }

                // ambient
                vec3 ambient = this->ambient;
                err = ini.get_vec3f("ambient", ambient.x, ambient.y, ambient.z);
                if (err == inifile::Ok)
                {
                    this->ambient = ambient;
                }

                // Initial camera
                std::string camera = initial_camera;
                err = ini.get_string("camera", camera, true /*remove quotes*/);
                if (err == inifile::Ok)
                {
                    initial_camera = camera;
                }

                // Screenshot file base name
                std::string screenshotbasename = screenshot_file_base;
                err = ini.get_string("screenshotbasename", screenshotbasename, true /*remove quotes*/);
                if (err == inifile::Ok)
                {
                    screenshot_file_base = screenshotbasename;
                }

                // bounces
                uint32_t bounces = this->bounces;
                err = ini.get_uint32("bounces", bounces);
                if (err == inifile::Ok)
                {
                    this->bounces = bounces;
                }

                // bvh
                std::string bvh = "";
                err = ini.get_string("bvh", bvh);
                if (err == inifile::Ok)
                {
                    if (bvh == "default")
                    {
                        build_strategy = Binned;
                    }
                    else if (bvh == "split")
                    {
                        build_strategy = Split;
                    }
                    else if (bvh == "lbvh")
                    {
                        build_strategy = LBVH;
                    }
                }

                // color space
                std::string colorspace = "";
                err = ini.get_string("colorspace", colorspace);
                if (err == inifile::Ok)
                {
                    if (colorspace == "rgb")
                    {
                        rt.color_space() = host_device_rt::RGB;
                    }
                    else
                    {
                        rt.color_space() = host_device_rt::SRGB;
                    }
                }

                // depth of field
                bool dof = use_dof;
                err = ini.get_bool("dof", dof);
                if (err == inifile::Ok)
                {
                    use_dof = dof;
                }

                // lens radius
                float lr = cam.get_lens_radius();
                err = ini.get_float("lens_radius", lr);
                if (err == inifile::Ok)
                {
                    cam.set_lens_radius(lr);
                }

                // focal distance
                float fd = cam.get_focal_distance();
                err = ini.get_float("focal_distance", fd);
                if (err == inifile::Ok)
                {
                    cam.set_focal_distance(fd);
                }

                // asynchronous rendering
                bool async = render_async;
                err = ini.get_bool("render_async", async);
                if (err == inifile::Ok)
                {
                    render_async = async;
                }

                // ImGui menu
                bool hud = show_hud;
                err = ini.get_bool("hud", hud);
                if (err == inifile::Ok)
                {
                    show_hud = hud;
                }

                // headlight
                bool headlight = use_headlight;
                err = ini.get_bool("headlight", headlight);
                if (err == inifile::Ok)
                {
                    use_headlight = headlight;
                }

                // ground plane
                bool groundplane = use_groundplane;
                err = ini.get_bool("groundplane", groundplane);
                if (err == inifile::Ok)
                {
                    use_groundplane = groundplane;
                }

                // Environment map
                std::string envmap = env_map_filename;
                err = ini.get_string("envmap", envmap, true /*remove quotes*/);
                if (err == inifile::Ok)
                {
                    env_map_filename = envmap;
                }

                // Supersampling
                uint32_t ssaa = spp;
                err = ini.get_uint32("ssaa", ssaa);
                if (err == inifile::Ok)
                {
                    spp = ssaa;
                }

                // Jittered supersampling (also manipulates this->spp!)
                uint32_t local_spp = spp;
                err = ini.get_uint32("spp", local_spp);
                if (err == inifile::Ok)
                {
                    spp = local_spp;
                }

#if VSNRAY_COMMON_HAVE_CUDA
                // Device (CPU or GPU)
                std::string device = "";
                err = ini.get_string("device", device);
                if (err == inifile::Ok)
                {
                    if (device == "cpu")
                    {
                        rt.mode() = host_device_rt::CPU;
                    }
                    else if (device == "gpu")
                    {
                        rt.mode() = host_device_rt::GPU;
                    }
                }
#endif

                // Don't consider other files
                break;
            }
        }
    }


    int                                         w               = 800;
    int                                         h               = 800;
    unsigned                                    frame_num       = 0;
    unsigned                                    bounces         = 0;
    unsigned                                    spp             = 1;
    algorithm                                   algo            = Simple;
    bvh_build_strategy                          build_strategy  = Binned;
    bool                                        use_headlight   = true;
    bool                                        use_groundplane = false;
    bool                                        use_dof         = false;
    bool                                        show_hud        = true;
    bool                                        show_bvh        = false;


    std::set<std::string>                       filenames;
    std::string                                 initial_camera;
    std::string                                 current_cam;
    std::string                                 screenshot_file_base = "screenshot";

    model                                       mod;
    vec3                                        ambient         = vec3(-1.0f);

    index_bvh<host_bvh_type::bvh_inst>          host_top_level_bvh;
    aligned_vector<host_bvh_type>               host_bvhs;
    aligned_vector<host_bvh_type::bvh_inst>     host_instances;
    aligned_vector<plastic<float>>              plastic_materials;
    aligned_vector<generic_material_t>          generic_materials;
    aligned_vector<point_light<float>>          point_lights;
    aligned_vector<spot_light<float>>           spot_lights;
    aligned_vector<area_light<float,
                   basic_triangle<3, float>>>   area_lights;
#if VSNRAY_COMMON_HAVE_PTEX
    aligned_vector<ptex::face_id_t>             ptex_tex_coords;
    aligned_vector<ptex::texture>               ptex_textures;
#endif
#if VSNRAY_COMMON_HAVE_CUDA
    cuda_index_bvh<device_bvh_type::bvh_inst>   device_top_level_bvh;
    std::vector<device_bvh_type>                device_bvhs;
    thrust::device_vector<normal_type>          device_geometric_normals;
    thrust::device_vector<normal_type>          device_shading_normals;
    thrust::device_vector<tex_coord_type>       device_tex_coords;
    thrust::device_vector<plastic<float>>       device_plastic_materials;
    thrust::device_vector<generic_material_t>   device_generic_materials;
    thrust::device_vector<color_type>           device_colors;
    std::map<std::string, device_tex_type>      device_texture_map;
    thrust::device_vector<device_tex_ref_type>  device_textures;
#endif

    host_sched_t<ray_type_cpu>                  host_sched;
    host_device_rt                              rt;
#if VSNRAY_COMMON_HAVE_CUDA
    cuda_sched<ray_type_gpu>                    device_sched;
#endif
    thin_lens_camera                            cam;

    std::string                                 env_map_filename;
    visionaray::texture<vec4, 2>                env_map;
    host_environment_light                      env_light;
#if VSNRAY_COMMON_HAVE_CUDA
    visionaray::cuda_texture<vec4, 2>           device_env_map;
    device_environment_light                    device_env_light;
#endif

    // List of cameras, e.g. read from scene graph
    aligned_vector<std::pair<
            std::string, thin_lens_camera>>     cameras;

    mouse::pos                                  mouse_pos;

    texture_format                              tex_format = UV;

    visionaray::frame_counter                   counter;
    double                                      last_frame_time = 0.0;
    bvh_outline_renderer                        outlines;
    gl::debug_callback                          gl_debug_callback;

    // Control if new path tracer convergence frames are accumulated
    bool                                        paused = false;

    // Number of path tracer convergece frames to be rendered (default: inf)
    unsigned                                    frames = unsigned(-1);

    bool                                        render_async  = false;
    std::future<void>                           render_future;
    std::mutex                                  display_mutex;


    static const std::string camera_file_base;
    static const std::string camera_file_suffix;

    void build_scene();

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_space_mouse_move(visionaray::space_mouse_event const& event);
    void on_resize(int w, int h);

private:

    void load_camera(std::string filename);
    void init_bvh_outlines();
    void clear_frame();
    void screenshot();
    void render_hud();
    void render_impl();

};

const std::string renderer::camera_file_base = "visionaray-camera";
const std::string renderer::camera_file_suffix = ".txt";


//-------------------------------------------------------------------------------------------------
// Map obj material to generic material
//

inline generic_material_t map_material(sg::obj_material const& mat)
{
    // Add emissive material if emissive component > 0
    if (length(mat.ce) > 0.0f)
    {
        emissive<float> em;
        em.ce() = from_rgb(mat.ce);
        em.ls() = 1.0f;
        return em;
    }
    else if (mat.illum == 1)
    {
        matte<float> ma;
        ma.ca() = from_rgb(mat.ca);
        ma.cd() = from_rgb(mat.cd);
        ma.ka() = 1.0f;
        ma.kd() = 1.0f;
        return ma;
    }
    else if (mat.illum == 3)
    {
        mirror<float> mi;
        mi.cr() = from_rgb(mat.cs);
        mi.kr() = 1.0f;
        mi.ior() = spectrum<float>(0.0f);
        mi.absorption() = spectrum<float>(0.0f);
        return mi;
    }
    else if (mat.illum == 4 && mat.transmission > 0.0f)
    {
        glass<float> gl;
        gl.ct() = from_rgb(mat.cd);
        gl.kt() = 1.0f;
        gl.cr() = from_rgb(mat.cs);
        gl.kr() = 1.0f;
        gl.ior() = from_rgb(mat.ior);
        return gl;
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
        return pl;
    }
}


//-------------------------------------------------------------------------------------------------
// Map disney material to generic material
//

inline generic_material_t map_material(sg::disney_material const& mat)
{
    // TODO..
    if (mat.refractive > 0.0f)
    {
        glass<float> gl;
        gl.ct() = from_rgb(mat.base_color.xyz());
        gl.kt() = 1.0f;
        gl.cr() = from_rgb(vec3(mat.spec_trans));
        gl.kr() = 1.0f;
        gl.ior() = from_rgb(vec3(mat.ior));
        return gl;
    }
    else
    {
        matte<float> ma;
        ma.ca() = from_rgb(vec3(0.0f));
        ma.cd() = from_rgb(vec3(mat.base_color.xyz()));
        ma.ka() = 1.0f;
        ma.kd() = 1.0f;
        return ma;
    }
}


//-------------------------------------------------------------------------------------------------
// Map metal material to generic material
//

inline generic_material_t map_material(sg::metal_material const& mat)
{
    // TODO: consolidate glass and sg::glass_material (?)
    metal<float> mt;
    mt.roughness() = mat.roughness;
    mt.absorption() = from_rgb(mat.absorption);
    mt.ior() = from_rgb(mat.ior);
    return mt;
}


//-------------------------------------------------------------------------------------------------
// Map glass material to generic material
//

inline generic_material_t map_material(sg::glass_material const& mat)
{
    // TODO: consolidate glass and sg::glass_material (?)
    glass<float> gl;
    gl.ct() = from_rgb(mat.ct);
    gl.kt() = 1.0f;
    gl.cr() = from_rgb(mat.cr);
    gl.kr() = 1.0f;
    gl.ior() = from_rgb(mat.ior);
    return gl;
}


//-------------------------------------------------------------------------------------------------
// I/O utility for camera lookat only - not fit for the general case!
//

inline std::istream& operator>>(std::istream& in, pinhole_camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

inline std::ostream& operator<<(std::ostream& out, pinhole_camera const& cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}


//-------------------------------------------------------------------------------------------------
// Approximate sphere with icosahedron
// cf. https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/
//

struct icosahedron
{
    aligned_vector<basic_triangle<3, float>> triangles;
    aligned_vector<vec3> normals;
};

inline icosahedron make_icosahedron()
{
    static constexpr float X = 0.525731112119133606f;
    static constexpr float Z = 0.850650808352039932f;
    static constexpr float N = 0.0f;

    static const vec3 vertices[] = {
        { -X,  N,  Z },
        {  X,  N,  Z },
        { -X,  N, -Z },
        {  X,  N, -Z },
        {  N,  Z,  X },
        {  N,  Z, -X },
        {  N, -Z,  X },
        {  N, -Z, -X },
        {  Z,  X,  N },
        { -Z,  X,  N },
        {  Z, -X,  N },
        { -Z, -X,  N }
        };

    static const vec3i indices[] {
        { 0, 4, 1 },
        { 0, 9, 4 },
        { 9, 5, 4 },
        { 4, 5, 8 },
        { 4, 8, 1 },
        { 8, 10, 1 },
        { 8, 3, 10 },
        { 5, 3, 8 },
        { 5, 2, 3 },
        { 2, 7, 3 },
        { 7, 10, 3 },
        { 7, 6, 10 },
        { 7, 11, 6 },
        { 11, 0, 6 },
        { 0, 1, 6 },
        { 6, 1, 10 },
        { 9, 0, 11 },
        { 9, 11, 2 },
        { 9, 2, 5 },
        { 7, 2, 11 }
        };

    auto make_triangle = [&](int index)
    {
        vec3i idx = indices[index];
        return basic_triangle<3, float>(
                vertices[idx.x],
                vertices[idx.y] - vertices[idx.x],
                vertices[idx.z] - vertices[idx.x]
                );
    };

    icosahedron result;
    result.triangles.resize(20);
    result.normals.resize(20 * 3);

    for (int i = 0; i < 20; ++i)
    {
        result.triangles[i] = make_triangle(i);

        vec3i idx = indices[i];

        vec3 v1 = vertices[idx.x];
        vec3 v2 = vertices[idx.y];
        vec3 v3 = vertices[idx.z];

        result.normals[i * 3] = normalize(v1);
        result.normals[i * 3 + 1] = normalize(v2);
        result.normals[i * 3 + 2] = normalize(v3);
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Reset triangle mesh flags to 0
//

struct reset_flags_visitor : sg::node_visitor
{
    using node_visitor::apply;

    void apply(sg::surface_properties& sp)
    {
        sp.flags() = 0;

        node_visitor::apply(sp);
    }

    void apply(sg::sphere& sph)
    {
        sph.flags() = 0;

        node_visitor::apply(sph);
    }

    void apply(sg::triangle_mesh& tm)
    {
        tm.flags() = 0;

        node_visitor::apply(tm);
    }

    void apply(sg::indexed_triangle_mesh& itm)
    {
        itm.flags() = 0;

        node_visitor::apply(itm);
    }
};


//-------------------------------------------------------------------------------------------------
// Instance
//

struct instance
{
    int index;
    mat4 transform;
};


//-------------------------------------------------------------------------------------------------
// Traverse the scene graph to construct geometry, materials and BVH instances
//

struct build_scene_visitor : sg::node_visitor
{
    using node_visitor::apply;

    build_scene_visitor(
            aligned_vector<renderer::host_bvh_type>& bvhs,
            aligned_vector<instance>& instances,
            aligned_vector<vec3>& shading_normals,
            aligned_vector<vec3>& geometric_normals,
            aligned_vector<vec2>& tex_coords,
            aligned_vector<vec3>& colors,
#if VSNRAY_COMMON_HAVE_PTEX
            aligned_vector<ptex::face_id_t>& face_ids,
#endif
            aligned_vector<std::pair<std::string, thin_lens_camera>>& cameras,
            aligned_vector<point_light<float>>& point_lights,
            aligned_vector<spot_light<float>>& spot_lights,
            visionaray::texture<vec4, 2>& env_map,
            host_environment_light& env_light,
            renderer::bvh_build_strategy build_strategy
            )
        : bvhs_(bvhs)
        , instances_(instances)
        , shading_normals_(shading_normals)
        , geometric_normals_(geometric_normals)
        , tex_coords_(tex_coords)
        , colors_(colors)
#if VSNRAY_COMMON_HAVE_PTEX
        , face_ids_(face_ids)
#endif
        , cameras_(cameras)
        , point_lights_(point_lights)
        , spot_lights_(spot_lights)
        , env_map_(env_map)
        , env_light_(env_light)
        , build_strategy_(build_strategy)
    {
    }

    void apply(sg::camera& c)
    {
        cameras_.push_back(std::make_pair(c.name(), static_cast<thin_lens_camera>(c)));

        node_visitor::apply(c);
    }

    void apply(sg::point_light& pl)
    {
        point_lights_.push_back(static_cast<visionaray::point_light<float>>(pl));

        vec3 pos = point_lights_.back().position();
        pos = (current_transform_ * vec4(pos, 1.0f)).xyz();
        point_lights_.back().set_position(pos);

        node_visitor::apply(pl);
    }

    void apply(sg::spot_light& sl)
    {
        spot_lights_.push_back(static_cast<visionaray::spot_light<float>>(sl));

        vec3 pos = spot_lights_.back().position();
        pos = (current_transform_ * vec4(pos, 1.0f)).xyz();
        spot_lights_.back().set_position(pos);

        node_visitor::apply(sl);
    }

    void apply(sg::environment_light& el)
    {
        auto tex = std::dynamic_pointer_cast<sg::texture2d<vec4>>(el.texture());

        if (tex != nullptr)
        {
            env_map_ = visionaray::texture<vec4, 2>(tex->width(), tex->height());
            env_map_.set_address_mode(tex->get_address_mode());
            env_map_.set_filter_mode(tex->get_filter_mode());
            env_map_.reset(tex->data());

            env_light_.texture() = texture_ref<vec4, 2>(env_map_);
            env_light_.scale() = from_rgb(el.scale());
            env_light_.set_light_to_world_transform(el.light_to_world_transform());
        }

        node_visitor::apply(el);
    }

    void apply(sg::transform& t)
    {
        mat4 prev = current_transform_;

        current_transform_ = current_transform_ * t.matrix();

        node_visitor::apply(t);

        current_transform_ = prev;
    }

    void apply(sg::surface_properties& sp)
    {
        unsigned prev = current_geom_id_;

        if (sp.flags() == 0 && sp.material() && sp.textures().find("diffuse") != sp.textures().end())
        {
            std::shared_ptr<sg::material> material = sp.material();
            std::shared_ptr<sg::texture> texture = sp.textures()["diffuse"];

            auto surf = std::make_pair(material, texture);

            auto it = std::find(surfaces.begin(), surfaces.end(), surf);
            if (it == surfaces.end())
            {
                current_geom_id_ = static_cast<unsigned>(surfaces.size());
                surfaces.push_back(surf);
            }
            else
            {
                current_geom_id_ = static_cast<unsigned>(std::distance(surfaces.begin(), it));
            }

            sp.flags() = ~sp.flags();
        }

        node_visitor::apply(sp);

        current_geom_id_ = prev;
    }

    void apply(sg::sphere& sph)
    {
        if (sph.flags() == 0)
        {
            auto ico = make_icosahedron();

            shading_normals_.insert(shading_normals_.end(), ico.normals.begin(), ico.normals.end());

            for (size_t i = 0; i < ico.triangles.size(); ++i)
            {
                auto& tri = ico.triangles[i];

                tri.prim_id = current_prim_id_++;
                tri.geom_id = current_geom_id_;

                vec3 n = normalize(cross(tri.e1, tri.e2));

                geometric_normals_.emplace_back(n);
                tex_coords_.emplace_back(0.0f, 0.0f);
                tex_coords_.emplace_back(0.0f, 0.0f);
                tex_coords_.emplace_back(0.0f, 0.0f);
            }

            // Build single bvh
            if (build_strategy_ == renderer::LBVH)
            {
                lbvh_builder builder;

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, ico.triangles.data(), ico.triangles.size()));
            }
            else
            {
                binned_sah_builder builder;
                builder.enable_spatial_splits(build_strategy_ == renderer::Split);

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, ico.triangles.data(), ico.triangles.size()));
            }

            sph.flags() = ~(bvhs_.size() - 1);
        }

        instances_.push_back({ static_cast<int>(~sph.flags()), current_transform_ });

        node_visitor::apply(sph);
    }

    void apply(sg::triangle_mesh& tm)
    {
        if (tm.flags() == 0 && tm.vertices.size() > 0)
        {
            assert(tm.vertices.size() % 3 == 0);

            aligned_vector<basic_triangle<3, float>> triangles(tm.vertices.size() / 3);

            size_t first_geometric_normal = geometric_normals_.size();
            geometric_normals_.resize(geometric_normals_.size() + tm.vertices.size() / 3);

            shading_normals_.insert(shading_normals_.end(), tm.normals.begin(), tm.normals.end());

            tex_coords_.insert(tex_coords_.end(), tm.tex_coords.begin(), tm.tex_coords.end());

            size_t first_color = colors_.size();
            if (tm.colors.size() > 0)
            {
                colors_.resize(first_color + tm.colors.size());
                for (size_t i = 0; i < tm.colors.size(); ++i)
                {
                    colors_[first_color + i] = vec3(tm.colors[i]);
                }
            }

#if VSNRAY_COMMON_HAVE_PTEX
            face_ids_.insert(face_ids_.end(), tm.face_ids.begin(), tm.face_ids.end());
#endif

            for (size_t i = 0; i < tm.vertices.size(); i += 3)
            {
                vec3 v1 = tm.vertices[i];
                vec3 v2 = tm.vertices[i + 1];
                vec3 v3 = tm.vertices[i + 2];

                basic_triangle<3, float> tri(v1, v2 - v1, v3 - v1);
                tri.prim_id = current_prim_id_++;
                tri.geom_id = current_geom_id_;
                triangles[i / 3] = tri;

                vec3 gn = normalize(cross(v2 - v1, v3 - v1));

                geometric_normals_[first_geometric_normal + i / 3] = gn;
            }

            // Build single bvh
            if (build_strategy_ == renderer::LBVH)
            {
                lbvh_builder builder;

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, triangles.data(), triangles.size()));
            }
            else
            {
                binned_sah_builder builder;
                builder.enable_spatial_splits(build_strategy_ == renderer::Split);

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, triangles.data(), triangles.size()));
            }

            tm.flags() = ~(bvhs_.size() - 1);
        }

        instances_.push_back({ static_cast<int>(~tm.flags()), current_transform_ });

        node_visitor::apply(tm);
    }

    void apply(sg::indexed_triangle_mesh& itm)
    {
        if (itm.flags() == 0 && itm.vertex_indices.size() > 0)
        {
            assert(itm.vertex_indices.size() % 3 == 0);

            aligned_vector<basic_triangle<3, float>> triangles(itm.vertex_indices.size() / 3);

            size_t first_geometric_normal = geometric_normals_.size();
            geometric_normals_.resize(geometric_normals_.size() + itm.vertex_indices.size() / 3);

            size_t first_shading_normal = shading_normals_.size();
            if (itm.normal_indices.size() > 0)
            {
                assert(itm.normal_indices.size() % 3 == 0);
                shading_normals_.resize(shading_normals_.size() + itm.normal_indices.size());
            }
            else
            {
                shading_normals_.resize(shading_normals_.size() + itm.vertex_indices.size());
            }

            size_t first_tex_coord = tex_coords_.size();
            if (itm.tex_coord_indices.size() > 0)
            {
                assert(itm.tex_coord_indices.size() % 3 == 0);
                tex_coords_.resize(tex_coords_.size() + itm.tex_coord_indices.size());
            }

            size_t first_color = colors_.size();
            if (itm.color_indices.size() > 0)
            {
                assert(itm.color_indices.size() % 3 == 0);
                colors_.resize(first_color + itm.color_indices.size());
            }

            for (size_t i = 0; i < itm.vertex_indices.size(); i += 3)
            {
                vec3 v1 = (*itm.vertices)[itm.vertex_indices[i]];
                vec3 v2 = (*itm.vertices)[itm.vertex_indices[i + 1]];
                vec3 v3 = (*itm.vertices)[itm.vertex_indices[i + 2]];

                basic_triangle<3, float> tri(v1, v2 - v1, v3 - v1);
                tri.prim_id = current_prim_id_++;
                tri.geom_id = current_geom_id_;
                triangles[i / 3] = tri;

                vec3 gn = normalize(cross(v2 - v1, v3 - v1));

                geometric_normals_[first_geometric_normal + i / 3] = gn;

                if (itm.normal_indices.size() == 0)
                {
                    shading_normals_[first_shading_normal + i]     = gn;
                    shading_normals_[first_shading_normal + i + 1] = gn;
                    shading_normals_[first_shading_normal + i + 2] = gn;
                }
            }

            for (size_t i = 0; i < itm.normal_indices.size(); i += 3)
            {
                shading_normals_[first_shading_normal + i]     = (*itm.normals)[itm.normal_indices[i]];
                shading_normals_[first_shading_normal + i + 1] = (*itm.normals)[itm.normal_indices[i + 1]];
                shading_normals_[first_shading_normal + i + 2] = (*itm.normals)[itm.normal_indices[i + 2]];
            }

            for (size_t i = 0; i < itm.tex_coord_indices.size(); i += 3)
            {
                tex_coords_[first_tex_coord + i]     = (*itm.tex_coords)[itm.tex_coord_indices[i]];
                tex_coords_[first_tex_coord + i + 1] = (*itm.tex_coords)[itm.tex_coord_indices[i + 1]];
                tex_coords_[first_tex_coord + i + 2] = (*itm.tex_coords)[itm.tex_coord_indices[i + 2]];
            }

            for (size_t i = 0; i < itm.color_indices.size(); i += 3)
            {
                colors_[first_color + i]     = vec3((*itm.colors)[itm.color_indices[i]]);
                colors_[first_color + i + 1] = vec3((*itm.colors)[itm.color_indices[i + 1]]);
                colors_[first_color + i + 2] = vec3((*itm.colors)[itm.color_indices[i + 2]]);
            }

#if VSNRAY_COMMON_HAVE_PTEX
            face_ids_.insert(face_ids_.end(), itm.face_ids.begin(), itm.face_ids.end());
#endif


            // Build single bvh
            if (build_strategy_ == renderer::LBVH)
            {
                lbvh_builder builder;

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, triangles.data(), triangles.size()));
            }
            else
            {
                binned_sah_builder builder;
                builder.enable_spatial_splits(build_strategy_ == renderer::Split);

                bvhs_.emplace_back(builder.build(renderer::host_bvh_type{}, triangles.data(), triangles.size()));
            }

            itm.flags() = ~(bvhs_.size() - 1);
        }

        instances_.push_back({ static_cast<int>(~itm.flags()), current_transform_ });

        node_visitor::apply(itm);
    }

    // List of surface properties to derive geom_ids from
    std::vector<std::pair<std::shared_ptr<sg::material>, std::shared_ptr<sg::texture>>> surfaces;

    // Current transform along the path
    mat4 current_transform_ = mat4::identity();


    // Storage bvhs
    aligned_vector<renderer::host_bvh_type>& bvhs_;

    // Instances (BVH index + transform)
    aligned_vector<instance>& instances_;

    // Shading normals
    aligned_vector<vec3>& shading_normals_;

    // Geometric normals
    aligned_vector<vec3>& geometric_normals_;

    // Texture coordinates
    aligned_vector<vec2>& tex_coords_;

    // Vertex colors
    aligned_vector<vec3>& colors_;

#if VSNRAY_COMMON_HAVE_PTEX
    // Ptex face ids
    aligned_vector<ptex::face_id_t>& face_ids_;
#endif

    // Cameras
    aligned_vector<std::pair<std::string, thin_lens_camera>>& cameras_;

    // Point lights
    aligned_vector<point_light<float>>& point_lights_;

    // Spot lights
    aligned_vector<spot_light<float>>& spot_lights_;

    // Environment map
    visionaray::texture<vec4, 2>& env_map_;

    // Environment light
    host_environment_light& env_light_;

    // Assign consecutive prim ids
    unsigned current_prim_id_ = 0;

    // Assign consecutive geom ids for each encountered material
    unsigned current_geom_id_ = 0;

    // BVH build strategy
    renderer::bvh_build_strategy build_strategy_;

};


//-------------------------------------------------------------------------------------------------
// Build up scene data structures
//

void renderer::build_scene()
{
//  timer t;

    std::cout << "Creating BVH...\n";

    if (mod.scene_graph == nullptr)
    {
        // Single BVH
        if (build_strategy == LBVH && rt.mode() == host_device_rt::CPU)
        {
            host_bvhs.resize(1);

            lbvh_builder builder;

            //timer t;
            host_bvhs[0] = builder.build(host_bvh_type{}, mod.primitives.data(), mod.primitives.size());
            //std::cout << t.elapsed() << '\n';
        }
#if VSNRAY_COMMON_HAVE_CUDA
        else if (build_strategy == LBVH && rt.mode() == host_device_rt::GPU)
        {
            device_bvhs.resize(1);

            lbvh_builder builder;

            //cuda::timer t;
            thrust::device_vector<primitive_type> primitives(mod.primitives);
            device_bvhs[0] = builder.build(device_bvh_type{}, thrust::raw_pointer_cast(primitives.data()), mod.primitives.size());
            //std::cout << t.elapsed() << '\n';
        }
#endif
        else
        {
            host_bvhs.resize(1);

            binned_sah_builder builder;
            builder.enable_spatial_splits(build_strategy == Split);

            //timer t;
            host_bvhs[0] = builder.build(host_bvh_type{}, mod.primitives.data(), mod.primitives.size());
            //std::cout << t.elapsed() << '\n';
        }

        if (!env_map_filename.empty() && boost::filesystem::exists(env_map_filename))
        {
            image img;
            if (img.load(env_map_filename))
            {
                env_map = visionaray::texture<vec4, 2>(img.width(), img.height());
                env_map.set_address_mode(Clamp);
                env_map.set_filter_mode(Linear);
                make_texture(env_map, img);

                env_light.texture() = texture_ref<vec4, 2>(env_map);
                env_light.scale() = from_rgb(vec3(1.0f));
                env_light.set_light_to_world_transform(mat4::identity());
            }

            // When we have an environment light , enforce the code path
            // with instances which will also support the light source
            host_instances.push_back(host_bvhs[0].inst(mat4x3(mat3::identity(), vec3(0.0))));

            // Any builder will suffice, we only have one instance..
            lbvh_builder builder;

            host_top_level_bvh = builder.build(
                    index_bvh<host_bvh_type::bvh_inst>{},
                    host_instances.data(),
                    host_instances.size()
                    );
        }
    }
    else
    {
        reset_flags_visitor reset_visitor;
        mod.scene_graph->accept(reset_visitor);

        aligned_vector<instance> instances;

        build_scene_visitor build_visitor(
                host_bvhs,
                instances,
                mod.shading_normals, // TODO!!!
                mod.geometric_normals,
                mod.tex_coords,
                mod.colors,
#if VSNRAY_COMMON_HAVE_PTEX
                ptex_tex_coords,
#endif
                cameras,
                point_lights,
                spot_lights,
                env_map,
                env_light,
                build_strategy
                );
        mod.scene_graph->accept(build_visitor);

        host_instances.resize(instances.size());
        for (size_t i = 0; i < instances.size(); ++i)
        {
            size_t index = instances[i].index;
            host_instances[i] = host_bvhs[index].inst(mat4x3(top_left(instances[i].transform), instances[i].transform(3).xyz()));
        }

        // Single BVH
        if (build_strategy == LBVH)
        {
            lbvh_builder builder;

            host_top_level_bvh = builder.build(
                    index_bvh<host_bvh_type::bvh_inst>{},
                    host_instances.data(),
                    host_instances.size()
                    );
        }
        else
        {
            binned_sah_builder builder;
            builder.enable_spatial_splits(false);

            host_top_level_bvh = builder.build(
                    index_bvh<host_bvh_type::bvh_inst>{},
                    host_instances.data(),
                    host_instances.size()
                    );
        }


        tex_format = renderer::UV;

#if VSNRAY_COMMON_HAVE_PTEX
        // Simply check the first texture of the first surface
        // Scene has either Ptex textures, or it doesn't
        if (build_visitor.surfaces.size() > 0
            && std::dynamic_pointer_cast<sg::ptex_texture>(build_visitor.surfaces[0].second) != nullptr)
        {
            tex_format = renderer::Ptex;
            ptex_textures.resize(build_visitor.surfaces.size());
        }
#endif

        // Insert dummy material (wavefront obj) if no surfaces
        // were parsed from sg
        if (build_visitor.surfaces.size() == 0)
        {
            build_visitor.surfaces.resize(1);

            // Material
            build_visitor.surfaces[0].first = std::make_shared<sg::obj_material>();

            // Texture
            vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
            auto tex = std::make_shared<sg::texture2d<vector<4, unorm<8>>>>();
            tex->resize(1, 1);
            tex->set_address_mode(Wrap);
            tex->set_filter_mode(Nearest);
            tex->reset(&dummy_texel);
            build_visitor.surfaces[0].second = tex;
        }

        if (tex_format == renderer::UV)
        {
            mod.textures.resize(build_visitor.surfaces.size());
        }

        for (size_t i = 0; i < build_visitor.surfaces.size(); ++i)
        {
            auto const& surf = build_visitor.surfaces[i];

            model::material_type newmat = {};

            if (auto disney = std::dynamic_pointer_cast<sg::disney_material>(surf.first))
            {
                generic_materials.emplace_back(map_material(*disney));
            }
            else if (auto obj = std::dynamic_pointer_cast<sg::obj_material>(surf.first))
            {
                generic_materials.emplace_back(map_material(*obj));
            }
            else if (auto metal = std::dynamic_pointer_cast<sg::metal_material>(surf.first))
            {
                generic_materials.emplace_back(map_material(*metal));
            }
            else if (auto glass = std::dynamic_pointer_cast<sg::glass_material>(surf.first))
            {
                generic_materials.emplace_back(map_material(*glass));
            }

#if VSNRAY_COMMON_HAVE_PTEX
            if (tex_format == renderer::Ptex)
            {
                auto ptex_tex = std::dynamic_pointer_cast<sg::ptex_texture>(surf.second);
                if (ptex_tex != nullptr)
                {
                    ptex_textures[i] = { ptex_tex->filename(), ptex_tex->cache() };
                }
            }
            else if (tex_format == renderer::UV)
#endif

            {
                auto tex = std::dynamic_pointer_cast<sg::texture2d<vector<4, unorm<8>>>>(surf.second);
                if (tex != nullptr)
                {
                    model::texture_type texture(tex->width(), tex->height());
                    texture.set_address_mode(tex->get_address_mode());
                    texture.set_filter_mode(tex->get_filter_mode());
                    texture.reset(tex->data());

                    auto it = mod.texture_map.insert(std::make_pair(tex->name(), std::move(texture)));
                    mod.textures[i] = model::texture_type::ref_type(it.first->second);
                }
            }
        }

        mod.bbox = host_top_level_bvh.node(0).get_bounds();
        mod.materials.push_back({});

#if 1
        mod.scene_graph.reset();
#endif
    }

//  std::cout << t.elapsed() << std::endl;
}

//-------------------------------------------------------------------------------------------------
// Load camera from file, reset frame counter and clear frame
//

void renderer::load_camera(std::string filename)
{
    std::ifstream file(filename);
    if (file.good())
    {
        file >> cam;
        counter.reset();
        clear_frame();
        std::cout << "Load camera from file: " << filename << '\n';
    }
}


//-------------------------------------------------------------------------------------------------
// Initialize BVH outline renderer
//

void renderer::init_bvh_outlines()
{
    outlines.destroy();

    if (rt.mode() == host_device_rt::CPU)
    {
        if (host_top_level_bvh.num_nodes() > 0)
        {
            outlines.init(host_top_level_bvh);
        }
        else
        {
            outlines.init(host_bvhs[0]);
        }
    }
#if VSNRAY_COMMON_HAVE_CUDA
    else if (rt.mode() == host_device_rt::GPU)
    {
        if (device_top_level_bvh.num_nodes() > 0)
        {
            index_bvh<host_bvh_type::bvh_inst> temp(device_top_level_bvh);
            outlines.init(temp);
        }
        else
        {
            host_bvh_type temp(device_bvhs[0]);
            outlines.init(temp);
        }
    }
#endif
}


//-------------------------------------------------------------------------------------------------
// If path tracing, clear frame buffer and reset frame counter
//

void renderer::clear_frame()
{
    if (render_future.valid() && render_async)
    {
        render_future.wait();
    }

    frame_num = 0;

    if (algo == Pathtracing)
    {
        rt.clear();
    }
}


//-------------------------------------------------------------------------------------------------
// Take a screenshot
//

void renderer::screenshot()
{
#if VSNRAY_COMMON_HAVE_PNG
    static const std::string screenshot_file_suffix = ".png";
    image::save_option opt1;
#else
    static const std::string screenshot_file_suffix = ".pnm";
    image::save_option opt1({"binary", true});
#endif

    // Swizzle to RGB8 for compatibility with pnm image
    std::vector<vector<3, unorm<8>>> rgb(rt.width() * rt.height());
    swizzle(
        rgb.data(),
        PF_RGB8,
        rt.color(),
        PF_RGBA32F,
        rt.width() * rt.height(),
        TruncateAlpha
        );

    if (rt.color_space() == host_device_rt::SRGB)
    {
        for (int y = 0; y < rt.height(); ++y)
        {
            for (int x = 0; x < rt.width(); ++x)
            {
                auto& color = rgb[y * rt.width() + x];
                color.x = powf(color.x, 1 / 2.2f);
                color.y = powf(color.y, 1 / 2.2f);
                color.z = powf(color.z, 1 / 2.2f);
            }
        }
    }

    // Flip so that origin is (top|left)
    std::vector<vector<3, unorm<8>>> flipped(rt.width() * rt.height());

    for (int y = 0; y < rt.height(); ++y)
    {
        for (int x = 0; x < rt.width(); ++x)
        {
            int yy = rt.height() - y - 1;
            flipped[yy * rt.width() + x] = rgb[y * rt.width() + x];
        }
    }

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const*>(flipped.data())
        );

    int inc = 0;
    std::string inc_str = "";

    std::string filename = screenshot_file_base + inc_str + screenshot_file_suffix;

    while (boost::filesystem::exists(filename))
    {
        ++inc;
        inc_str = std::to_string(inc);

        while (inc_str.length() < 4)
        {
            inc_str = std::string("0") + inc_str;
        }

        inc_str = std::string("-") + inc_str;

        filename = screenshot_file_base + inc_str + screenshot_file_suffix;
    }

    if (img.save(filename, {opt1}))
    {
        std::cout << "Screenshot saved to file: " << filename << '\n';
    }
    else
    {
        std::cerr << "Error saving screenshot to file: " << filename << '\n';
    }
}


//-------------------------------------------------------------------------------------------------
// HUD
//

void renderer::render_hud()
{
    if (!have_imgui_support())
    {
        return;
    }

    // gather data to render

    int w = width();
    int h = height();

    int x = visionaray::clamp( mouse_pos.x, 0, w - 1 );
    int y = visionaray::clamp( mouse_pos.y, 0, h - 1 );
    auto color = rt.color();
    vec4 rgba(color[(h - 1 - y) * w + x]);

    int num_nodes = 0;
    int num_leaves = 0;

    float focal_dist = cam.get_focal_distance();
    float lens_radius = cam.get_lens_radius();

    if (!host_bvhs.empty())
    {
        traverse_depth_first(
            host_bvhs[0],
            [&](renderer::host_bvh_type::node_type const& node)
            {
                ++num_nodes;

                if (is_leaf(node))
                {
                    ++num_leaves;
                }
            }
            );
    }


    std::vector<std::string> camera_names;

    camera_names.push_back("<reset...>");


    // Cameras from scene graph
    for (auto& c : cameras)
    {
        camera_names.push_back(c.first);
    }


    // Camera files
    int inc = 0;
    std::string inc_str = "";

    std::string filename = camera_file_base + inc_str + camera_file_suffix;

    while (boost::filesystem::exists(filename))
    {
        camera_names.push_back(filename);

        ++inc;
        inc_str = std::to_string(inc);

        while (inc_str.length() < 4)
        {
            inc_str = std::string("0") + inc_str;
        }

        inc_str = std::string("-") + inc_str;

        filename = camera_file_base + inc_str + camera_file_suffix;
    }


    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Settings", &show_hud);

    std::array<char const*, 4> algo_names = {{
            "Simple",
            "Whitted",
            "Path Tracing",
            "Costs"
            }};

    std::array<char const*, 4> ssaa_modes = {{
            "1x SSAA",
            "2x SSAA",
            "4x SSAA",
            "8x SSAA"
            }};

    if (ImGui::BeginTabBar("Menu"))
    {
        // Overview tab
        if (ImGui::BeginTabItem("Overview"))
        {
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
            ImGui::Spacing();
            ImGui::Text("FPS: %6.2f", last_frame_time);
            ImGui::SameLine();
            ImGui::Spacing();
            ImGui::SameLine();
            if (algo == Pathtracing)
            {
                ImGui::Text("Frames: %7u", std::max(1U, frame_num));
            }
            else
            {
                ImGui::Text("SSAA: %dx", spp);
            }

            ImGui::Text("Device: %s", rt.mode() == host_device_rt::GPU ? "GPU" : "CPU");
            ImGui::EndTabItem();
        }

        // Settings tab
        if (ImGui::BeginTabItem("Settings"))
        {
            ImGui::Spacing();
            if (ImGui::Checkbox("Headlight", &use_headlight))
            {
                clear_frame();
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Render async", &render_async))
            {
                if (render_future.valid() && !render_async)
                {
                    render_future.wait();
                }
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("BVH", &show_bvh))
            {
                if (show_bvh)
                {
                    init_bvh_outlines();
                }
            }

            bool gamma = rt.color_space() == host_device_rt::SRGB;
            ImGui::Checkbox("Color space:", &gamma);
            ImGui::SameLine();
            if (gamma)
            {
                rt.color_space() = host_device_rt::SRGB;
                ImGui::Text("SRGB");
            }
            else
            {
                rt.color_space() = host_device_rt::RGB;
                ImGui::Text(" RGB");
            }
            ImGui::SameLine();
            int bounces = this->bounces;
            ImGui::PushItemWidth(80);
            if (ImGui::InputInt("Bounces", &bounces))
            {
                this->bounces = static_cast<unsigned>(bounces);
                counter.reset();
                clear_frame();
            }
            ImGui::PopItemWidth();

            int nbit = sizeof(spp) * 8;
            int ssaa_log2 = nbit - 1 - detail::clz(spp);
            std::string currentssaa = ssaa_modes[ssaa_log2];
            if (ImGui::BeginCombo("SPP", currentssaa.c_str()))
            {
                for (size_t i = 0; i < ssaa_modes.size(); ++i)
                {
                    bool selectedssaa = currentssaa == ssaa_modes[i];

                    if (ImGui::Selectable(ssaa_modes[i], selectedssaa))
                    {
                        currentssaa = ssaa_modes[i];
                        if (ssaa_modes[i] == ssaa_modes[0])
                        {
                            spp = 1;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if (ssaa_modes[i] == ssaa_modes[1])
                        {
                            spp = 2;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if (ssaa_modes[i] == ssaa_modes[2])
                        {
                            spp = 4;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if ( ssaa_modes[i] == ssaa_modes[3])
                        {

                            spp = 8;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Spacing();
            vec3 amb = ambient.x < 0.0f ? vec3(0.0f) : ambient;
            if (ImGui::ColorEdit4("Ambient Color", amb.data(), ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float))
            {
                ambient = amb;
                clear_frame();
            }

            ImGui::Spacing();
            ImGui::Spacing();
            char const* currentalgo = algo_names[algo];

            if (ImGui::BeginCombo("Algorithm", currentalgo))
            {
                for (size_t i = 0; i < algo_names.size(); ++i)
                {
                    bool selected = currentalgo == algo_names[i];

                    if (ImGui::Selectable(algo_names[i], selected))
                    {
                        currentalgo = algo_names[i];
                        if (i == 0)
                        {
                            rt.set_double_buffering(true);
                            algo = Simple;
                            counter.reset();
                            clear_frame();
                        }
                        else if (i == 1)
                        {
                            rt.set_double_buffering(true);
                            algo = Whitted;
                            counter.reset();
                            clear_frame();
                        }
                        else if (i == 2)
                        {
                            if (render_future.valid())
                            {
                                render_future.wait();
                            }
                            // Double buffering does not work in case of pathtracing
                            // because destination and source buffers need to be the same
                            rt.set_double_buffering(false);
                            algo = Pathtracing;
                            counter.reset();
                            clear_frame();
                        }
                        else if (i == 3)
                        {
                            rt.set_double_buffering(true);
                            algo = Costs;
                            counter.reset();
                            clear_frame();
                        }
                    }

                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                ImGui::EndCombo();
            }
            ImGui::EndTabItem();
        }

        // Camera tab
        if (ImGui::BeginTabItem("Camera"))
        {

            ImGui::Spacing();

            if (current_cam == "")
            {
                current_cam = camera_names[0]; // <reset...>
            }

            if (ImGui::BeginCombo("Cameras", current_cam.c_str()))
            {
                for (size_t i = 0; i < camera_names.size(); ++i)
                {
                    bool selected = current_cam == camera_names[i];

                    if (ImGui::Selectable(camera_names[i].c_str(), selected))
                    {
                        current_cam = camera_names[i].c_str();

                        auto it = std::find_if(
                                cameras.begin(),
                                cameras.end(),
                                [&](std::pair<std::string, thin_lens_camera> c)
                                {
                                    return c.first == camera_names[i];
                                }
                                );

                        if (it != cameras.end())
                        {
                            // Preloaded
                            cam = it->second;

                            // Reset viewport and aspect
                            float fovy = cam.fovy();
                            float aspect = width() / static_cast<float>(height());
                            cam.set_viewport(0, 0, width(), height());
                            cam.perspective(fovy, aspect, 0.001f, 1000.0f);

                            counter.reset();
                            clear_frame();
                        }
                        else if (current_cam == camera_names[0])
                        {
                            // Reset...
                            float aspect = width() / static_cast<float>(height());

                            cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
                            cam.set_lens_radius(0.1f);
                            cam.set_focal_distance(10.0f);
                            cam.view_all(mod.bbox);

                            counter.reset();
                            clear_frame();
                        }
                        else if (boost::filesystem::exists(current_cam))
                        {
                            // From file
                            load_camera(current_cam);
                        }
                    }

                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                ImGui::EndCombo();
            }
            ImGui::Spacing();
            int bc = 0;
            if (ImGui::Button("Save Cam"))
            {
                bc = 1;
            }

            if (bc == 1)
            {
                int inc = 0;
                std::string inc_str = "";

                std::string filename = camera_file_base + inc_str + camera_file_suffix;

                while (boost::filesystem::exists(filename))
                {
                    ++inc;
                    inc_str = std::to_string(inc);

                    while (inc_str.length() < 4)
                    {
                        inc_str = std::string("0") + inc_str;
                    }

                    inc_str = std::string("-") + inc_str;

                    filename = camera_file_base + inc_str + camera_file_suffix;
                }

                std::ofstream file(filename);
                if (file.good())
                {
                    std::cout << "Storing camera to file: " << filename << '\n';
                    file << cam;
                }
                bc = 0;
            }
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::PushItemWidth(50);
            if (ImGui::InputFloat("Lens radius", &lens_radius))
            {
                cam.set_lens_radius(lens_radius);
                if (use_dof && algo == Pathtracing)
                {
                    clear_frame();
                }

                if (algo != Pathtracing)
                {
                    std::cerr << "Warning: setting only affects pathtracing algorithm\n";
                }
            }
            ImGui::PopItemWidth();
            ImGui::Spacing();
            ImGui::Spacing();
            if (ImGui::Checkbox("DoF", &use_dof))
            {
                if (use_dof && algo == Pathtracing) 
                {
                    clear_frame();
                }

                if (algo != Pathtracing)
                {
                    std::cerr << "Warning: setting only affects pathtracing algorithm\n";
                }
            }
            ImGui::SameLine();
            ImGui::PushItemWidth(200);
            if (ImGui::SliderFloat("##Focal", &focal_dist, 0.1, 100.0f, "Focal Dist. %.1f"))
            {
                cam.set_focal_distance(focal_dist);
                if (use_dof && algo == Pathtracing)
                {
                    clear_frame();
                }

                if (algo != Pathtracing)
                {
                    std::cerr << "Warning: setting only affects pathtracing algorithm\n";
                }
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::SameLine(ImGui::GetWindowWidth()-39);
    ImGui::SetCursorPosY(25);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(7.0f, 7.6f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(1.0f, 7.6f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(7.0f, 0.6f, 0.6f));
    if (ImGui::Button("Quit"))
    {
        quit();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::End();
}

void renderer::render_impl()
{
    if (paused)
    {
        return;
    }

    if (use_headlight)
    {
        point_light<float> headlight;
        headlight.set_cl( vec3(1.0, 1.0, 1.0) );
        headlight.set_kl(1.0);
        headlight.set_position( cam.eye() );
        headlight.set_constant_attenuation(1.0);
        headlight.set_linear_attenuation(0.0);
        headlight.set_quadratic_attenuation(0.0);
        point_lights.push_back(headlight);
    }

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = this->bounces ? this->bounces : algo == Pathtracing ? 10U : 4U;
    auto epsilon    = std::max( 1E-3f, length(diagonal) * 1E-5f );
    auto amb        = ambient.x >= 0.0f // if set via cmdline
                            ? vec4(ambient, 1.0f)
                            : vec4(0.0)
                            ;

    camera_t camx = cam;
    if (!use_dof || algo != Pathtracing)
    {
        camx.set_lens_radius(0.0f);
    }

    if (rt.mode() == host_device_rt::CPU)
    {
        if (host_top_level_bvh.num_primitives() > 0)
        {
            aligned_vector<generic_light_t> temp_lights;
            for (auto pl : point_lights)
            {
                temp_lights.push_back(pl);
            }

            for (auto sl : spot_lights)
            {
                temp_lights.push_back(sl);
            }

            for (auto al : area_lights)
            {
                temp_lights.push_back(al);
            }
            if (tex_format == renderer::UV)
            {
                render_instances_cpp(
                        host_top_level_bvh,
                        mod.geometric_normals,
                        mod.shading_normals,
                        mod.tex_coords,
                        generic_materials,
                        mod.colors,
                        mod.textures,
                        temp_lights,
                        bounces,
                        epsilon,
                        vec4(background_color(), 1.0f),
                        amb,
                        rt,
                        host_sched,
                        camx,
                        frame_num,
                        algo,
                        spp,
                        env_light
                        );
            }
#if VSNRAY_COMMON_HAVE_PTEX
            else if (tex_format == renderer::Ptex)
            {
                render_instances_ptex_cpp(
                        host_top_level_bvh,
                        mod.geometric_normals,
                        mod.shading_normals,
                        ptex_tex_coords,
                        generic_materials,
                        mod.colors,
                        ptex_textures,
                        temp_lights,
                        bounces,
                        epsilon,
                        vec4(background_color(), 1.0f),
                        amb,
                        rt,
                        host_sched,
                        camx,
                        frame_num,
                        algo,
                        spp,
                        env_light
                        );
            }
#endif
        }
        else if (area_lights.size() > 0 && algo == Pathtracing)
        {
            render_generic_material_cpp(
                    host_bvhs[0],
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
                    camx,
                    frame_num,
                    algo,
                    spp
                    );
        }
        else
        {
            render_plastic_cpp(
                    host_bvhs[0],
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
                    camx,
                    frame_num,
                    algo,
                    spp
                    );
        }
    }
#if VSNRAY_COMMON_HAVE_CUDA
    else if (rt.mode() == host_device_rt::GPU)
    {
        if (device_top_level_bvh.num_primitives() > 0)
        {
            // TODO: disambiguate
            aligned_vector<generic_light_t> temp_lights;
            for (auto pl : point_lights)
            {
                temp_lights.push_back(pl);
            }

            for (auto sl : spot_lights)
            {
                temp_lights.push_back(sl);
            }

            for (auto al : area_lights)
            {
                temp_lights.push_back(al);
            }

            if (tex_format == renderer::UV)
            {
                render_instances_cu(
                        device_top_level_bvh,
                        device_geometric_normals,
                        device_shading_normals,
                        device_tex_coords,
                        device_generic_materials,
                        device_colors,
                        device_textures,
                        temp_lights,
                        bounces,
                        epsilon,
                        vec4(background_color(), 1.0f),
                        amb,
                        rt,
                        device_sched,
                        camx,
                        frame_num,
                        algo,
                        spp,
                        device_env_light
                        );
            }
        }
        else if (area_lights.size() > 0 && algo == Pathtracing)
        {
            render_generic_material_cu(
                    device_bvhs[0],
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
                    camx,
                    frame_num,
                    algo,
                    spp
                    );
        }
        else
        {
            render_plastic_cu(
                    device_bvhs[0],
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
                    camx,
                    frame_num,
                    algo,
                    spp
                    );
        }
    }
#endif

    last_frame_time = counter.register_frame();

#if VSNRAY_COMMON_HAVE_PTEX
    if (ptex_textures.size() > 0)
    {
        PtexCache::Stats stats;
        ptex_textures[0].cache.get()->get()->getStats(stats);
        std::cout << "Mem used:        " << stats.memUsed << '\n';
        std::cout << "Peak mem used:   " << stats.peakMemUsed << '\n';
        std::cout << "Files open:      " << stats.filesOpen << '\n';
        std::cout << "Peak files open: " << stats.peakFilesOpen << '\n';
        std::cout << "Files accessed:  " << stats.filesAccessed << '\n';
        std::cout << "File reopens:    " << stats.fileReopens << '\n';
        std::cout << "Block reads:     " << stats.blockReads << '\n';
    }
#endif

    if (use_headlight)
    {
        point_lights.erase(point_lights.end() - 1);
    }

    if (frames != unsigned(-1) && frame_num == frames)
    {
        if (!paused)
        {
            screenshot();
        }

        paused = true;
    }
}

void renderer::on_close()
{
    outlines.destroy();
}

void renderer::on_display()
{
    if (render_async)
    {
        if (!render_future.valid() || render_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            render_future = std::async(
                    std::launch::async,
                    [this]()
                    {
                        render_impl();

                        std::unique_lock<std::mutex> l(display_mutex);
                        rt.swap_buffers();
                    }
                    );
        }

        if (rt.width() == width() && rt.height() == height())
        {
            std::unique_lock<std::mutex> l(display_mutex);
            rt.display_color_buffer();
        }
    }
    else
    {
        render_impl();

        rt.swap_buffers();

        rt.display_color_buffer();
    }


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
    switch (event.key())
    {
    case '1':
        std::cout << "Switching algorithm: simple\n";
        if (!rt.get_double_buffering())
        {
            rt.set_double_buffering(true);

            // Make sure that 2nd buffer is allocated!
            rt.resize(width(), height());
        }
        algo = Simple;
        counter.reset();
        clear_frame();
        break;

    case '2':
        std::cout << "Switching algorithm: whitted\n";
        if (!rt.get_double_buffering())
        {
            rt.set_double_buffering(true);

            // Make sure that 2nd buffer is allocated!
            rt.resize(width(), height());
        }
        algo = Whitted;
        counter.reset();
        clear_frame();
        break;

    case '3':
        std::cout << "Switching algorithm: path tracing\n";
        if (render_future.valid())
        {
            render_future.wait();
        }
        // Double buffering does not work in case of pathtracing
        // because destination and source buffers need to be the same
        rt.set_double_buffering(false);
        algo = Pathtracing;
        counter.reset();
        clear_frame();
        break;

    case '4':
        std::cout << "Switching algorithm: BVH cost debugging\n";
        if (!rt.get_double_buffering())
        {
            rt.set_double_buffering(true);

            // Make sure that 2nd buffer is allocated!
            rt.resize(width(), height());
        }
        algo = Costs;
        counter.reset();
        clear_frame();
        break;

    case 'b':
        show_bvh = !show_bvh;

        if (show_bvh)
        {
            init_bvh_outlines();
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

    case 'l':
        use_headlight = !use_headlight;
        counter.reset();
        clear_frame();
        break;

   case 'm':
#if VSNRAY_COMMON_HAVE_CUDA
        if (rt.mode() == host_device_rt::CPU)
        {
            rt.mode() = host_device_rt::GPU;

            if (device_bvhs.empty())
            {
                copy_bvhs(
                    device_bvhs,
                    device_top_level_bvh,
                    host_bvhs,
                    host_top_level_bvh,
                    copy_kind::HostToDevice
                    );
            }
        }
        else
        {
            rt.mode() = host_device_rt::CPU;

            if (host_bvhs.empty())
            {
                copy_bvhs(
                    host_bvhs,
                    host_top_level_bvh,
                    device_bvhs,
                    device_top_level_bvh,
                    copy_kind::DeviceToHost
                    );
            }
        }
        counter.reset();
        clear_frame();

        if (show_bvh)
        {
            init_bvh_outlines();
        }
#endif
        break;

    case 'p':
        screenshot();
        break;

    case 's':
        spp *= 2;
        if (spp > 8)
        {
            spp = 1;
        }

        if (algo != Pathtracing)
        {
            counter.reset();
            clear_frame();
        }
        break;

    case 'u':
        {
            int inc = 0;
            std::string inc_str = "";

            std::string filename = camera_file_base + inc_str + camera_file_suffix;

            while (boost::filesystem::exists(filename))
            {
                ++inc;
                inc_str = std::to_string(inc);

                while (inc_str.length() < 4)
                {
                    inc_str = std::string("0") + inc_str;
                }

                inc_str = std::string("-") + inc_str;

                filename = camera_file_base + inc_str + camera_file_suffix;
            }

            std::ofstream file(filename);
            if (file.good())
            {
                std::cout << "Storing camera to file: " << filename << '\n';
                file << cam;
            }
        }
        break;

    case 'v':
        {
            std::string filename = camera_file_base + camera_file_suffix;

            load_camera(filename);
        }
        break;

    case keyboard::Space:
        paused = !paused;

        if (paused)
        {
            std::cout << "Path tracer convergence paused\n";
        }
        else
        {
            std::cout << "Path tracer convergence resumed\n";
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

void renderer::on_space_mouse_move(visionaray::space_mouse_event const& event)
{
    clear_frame();

    viewer_type::on_space_mouse_move(event);
}

void renderer::on_resize(int w, int h)
{
    if (render_future.valid() && algo != Pathtracing)
    {
        render_future.wait();
    }

    cam.set_viewport(0, 0, w, h);
    float fovy = cam.fovy();
    float aspect = w / static_cast<float>(h);
    float z_near = cam.z_near();
    float z_far = cam.z_far();
    cam.perspective(fovy, aspect, z_near, z_far);
    rt.resize(w, h);
    clear_frame();
    viewer_type::on_resize(w, h);
}

int main(int argc, char** argv)
{
    renderer rend;

#if VSNRAY_COMMON_HAVE_CUDA
    int device = 0;
    cudaDeviceProp prop;
    if (rend.rt.direct_rendering() && cudaChooseDevice(&device, &prop) != cudaSuccess)
    {
        std::cerr << "Cannot choose CUDA device " << device << '\n';
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

	if (rend.filenames.empty())
	{
		std::cout << rend.cmd_line_inst().help(argv[0]) << "\n";
		return EXIT_FAILURE;
	}

    if (rend.algo == Pathtracing)
    {
        // Double buffering does not work in case of pathtracing
        // because destination and source buffers need to be the same
        rend.rt.set_double_buffering(false);
    }

    rend.gl_debug_callback.activate();

    // Load the scene
    std::cout << "Loading model...\n";

    std::vector<std::string> filenames;
    std::copy(rend.filenames.begin(), rend.filenames.end(), std::back_inserter(filenames));

    if (!rend.mod.load(filenames))
    {
        std::cerr << "Failed loading model\n";
        return EXIT_FAILURE;
    }

    if (rend.use_groundplane)
    {
        vec3 size = rend.mod.bbox.size();
        vec3 size2 = rend.mod.bbox.size() / 2.0f;

        float max_len = max_element(size);

        vec3 slack(1.5f, 1.0f, 1.5f);

        vec3 v1 = rend.mod.bbox.center() + vec3(-max_len / 2.0f, -size2.y, -max_len / 2.0f) * slack;
        vec3 v2 = rend.mod.bbox.center() + vec3(+max_len / 2.0f, -size2.y, -max_len / 2.0f) * slack;
        vec3 v3 = rend.mod.bbox.center() + vec3(+max_len / 2.0f, -size2.y, +max_len / 2.0f) * slack;
        vec3 v4 = rend.mod.bbox.center() + vec3(-max_len / 2.0f, -size2.y, +max_len / 2.0f) * slack;

        basic_triangle<3, float> t1;
        t1.v1 = v1;
        t1.e1 = v2 - v1;
        t1.e2 = v3 - v1;
        t1.prim_id = static_cast<unsigned>(rend.mod.primitives.size());
        t1.geom_id = static_cast<unsigned>(rend.mod.materials.size());
        rend.mod.primitives.push_back(t1);

        basic_triangle<3, float> t2;
        t2.v1 = v1;
        t2.e1 = v3 - v1;
        t2.e2 = v4 - v1;
        t2.prim_id = static_cast<unsigned>(rend.mod.primitives.size());
        t2.geom_id = static_cast<unsigned>(rend.mod.materials.size());
        rend.mod.primitives.push_back(t2);

        rend.mod.geometric_normals.push_back({ 0.0f, 1.0f, 0.0f });
        rend.mod.geometric_normals.push_back({ 0.0f, 1.0f, 0.0f });

        if (!rend.mod.shading_normals.empty())
        {
            for (int i = 0; i < 6; ++i)
            {
                rend.mod.shading_normals.push_back({ 0.0f, 1.0f, 0.0f });
            }
        }

        // Default obj material
        sg::obj_material mat;
        mat.cs = vec3(0.0f);
        rend.mod.materials.push_back(mat);

        // Dummy texture
        using tex_type = model::texture_type;

        tex_type tex(1, 1);
        tex.set_address_mode(Wrap);
        tex.set_filter_mode(Nearest);

        vector<4, unorm<8>> dummy_texel(1.0f, 1.0f, 1.0f, 1.0f);
        tex.reset(&dummy_texel);

        rend.mod.texture_map.insert(std::make_pair("null", std::move(tex)));

        // Maybe a "null" texture was already present and thus not inserted
        //  ==> find the one that was already inserted
        auto it = rend.mod.texture_map.find("null");

        // Insert a ref
        rend.mod.textures.push_back(tex_type::ref_type(it->second));
    }

    rend.build_scene();

    // Generate a list with plastic materials
    rend.plastic_materials = make_materials(
            plastic<float>{},
            rend.mod.materials
            );

    if (rend.generic_materials.empty())
    {
        // Generate another list with generic materials
        rend.generic_materials = make_materials(
                generic_material_t{},
                rend.mod.materials,
                [](aligned_vector<generic_material_t>& cont, model::material_type mat)
                {
                    cont.emplace_back(map_material(mat));
                }
                );
    }


    // Loop over all triangles, check if their
    // material is emissive, and if so, build
    // BVHs to create area lights from.

    struct range
    {
        std::size_t begin;
        std::size_t end;
        unsigned    bvh_id;
        unsigned    geom_id;
    };

    std::vector<range> ranges;

    for (unsigned b = 0; b < rend.host_bvhs.size(); ++b)
    {
        for (std::size_t i = 0; i < rend.host_bvhs[b].primitives().size(); ++i)
        {
            auto pi = rend.host_bvhs[b].primitives()[i];
            if (rend.generic_materials[pi.geom_id].as<emissive<float>>() != nullptr)
            {
                range r;
                r.begin = i;

                std::size_t j = i + 1;
                for (;j < rend.host_bvhs[b].primitives().size(); ++j)
                {
                    auto pii = rend.host_bvhs[b].primitives()[j];
                    if (rend.generic_materials[pii.geom_id].as<emissive<float>>() == nullptr
                                    || pii.geom_id != pi.geom_id)
                    {
                        break;
                    }
                }

                r.end = j;
                r.bvh_id = b;
                r.geom_id = pi.geom_id;
                ranges.push_back(r);

                i = r.end - 1;
            }
        }
    }

    // Build vector with area light sources
    for (auto r : ranges)
    {
        for (std::size_t i = r.begin; i != r.end; ++i)
        {
            area_light<float, basic_triangle<3, float>> light(rend.host_bvhs[r.bvh_id].primitives()[i]);
            auto mat = *rend.generic_materials[r.geom_id].as<emissive<float>>();
            light.set_cl(to_rgb(mat.ce()));
            light.set_kl(mat.ls());
            rend.area_lights.push_back(light);
        }
    }

    std::cout << "Ready\n";

#if VSNRAY_COMMON_HAVE_CUDA
    // Copy data to GPU
    try
    {
        if (rend.rt.mode() == host_device_rt::GPU && rend.build_strategy != renderer::LBVH)
        {
            copy_bvhs(
                rend.device_bvhs,
                rend.device_top_level_bvh,
                rend.host_bvhs,
                rend.host_top_level_bvh,
                copy_kind::HostToDevice
                );
        }

        // TODO: similar to BVH uploads, when we're not yet rendering on
        // the device, we could defer this, too!
        rend.device_geometric_normals = rend.mod.geometric_normals;
        rend.device_shading_normals = rend.mod.shading_normals;
        rend.device_tex_coords = rend.mod.tex_coords;
        rend.device_plastic_materials = rend.plastic_materials;
        rend.device_generic_materials = rend.generic_materials;
        rend.device_colors = rend.mod.colors;


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

        // Copy environment texture to the GPU
        if (rend.env_map)
        {
            rend.device_env_map = cuda_texture<vec4, 2>(rend.env_map);

            rend.device_env_light.texture() = cuda_texture_ref<vec4, 2>(rend.device_env_map);
            rend.device_env_light.scale() = rend.env_light.scale();
            rend.device_env_light.set_light_to_world_transform(rend.env_light.light_to_world_transform());
        }
    }
    catch (std::bad_alloc const&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        rend.device_bvhs.clear();
        rend.device_bvhs.shrink_to_fit();
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
        rend.device_colors.clear();
        rend.device_colors.shrink_to_fit();
        rend.device_texture_map.clear();
        rend.device_textures.clear();
        rend.device_textures.shrink_to_fit();
    }
#endif

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.set_lens_radius(0.1f);
    rend.cam.set_focal_distance(10.0f);

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
