// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
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

#include <boost/filesystem.hpp>

#include <imgui.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/gl/bvh_outline_renderer.h>
#include <visionaray/gl/debug_callback.h>
#include <visionaray/math/math.h>
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
#include <visionaray/spot_light.h>
#include <visionaray/thin_lens_camera.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/inifile.h>
#include <common/make_materials.h>
#include <common/model.h>
#include <common/sg.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <common/cuda.h>
#endif

#if VSNRAY_COMMON_HAVE_PTEX
#include <common/ptex.h>
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
#ifdef __CUDACC__
        , device_sched(8, 8)
#endif
        , mouse_pos(0)
    {
        using namespace support;

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
                { "split",              Split,          "Binned SAH with spatial splits" },
                { "lbvh",               LBVH,           "LBVH (CPU)" }
            },
            "bvh",
            cl::Desc("BVH build strategy"),
            cl::ArgRequired,
            cl::init(this->build_strategy)
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

        add_cmdline_option( cl::makeOption<bool&>(
            cl::Parser<>(),
            "headlight",
            cl::Desc("Activate headlight"),
            cl::ArgRequired,
            cl::init(this->use_headlight)
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

                // Supersampling
                uint32_t ssaa = ssaa_samples;
                err = ini.get_uint32("ssaa", ssaa);
                if (err == inifile::Ok)
                {
                    ssaa_samples = ssaa;
                }

#ifdef __CUDACC__
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
    unsigned                                    ssaa_samples    = 1;
    algorithm                                   algo            = Simple;
    bvh_build_strategy                          build_strategy  = Binned;
    bool                                        use_headlight   = true;
    bool                                        use_dof         = false;
    bool                                        show_hud        = true;
    bool                                        show_bvh        = false;


    std::set<std::string>                       filenames;
    std::string                                 initial_camera;
    std::string                                 current_cam;

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

    host_sched_t<ray_type_cpu>                  host_sched;
    host_device_rt                              rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>                    device_sched;
#endif
    thin_lens_camera                            cam;

    std::shared_ptr<visionaray::texture<vec4, 2>>
                                                environment_map = nullptr;


    // List of cameras, e.g. read from scene graph
    aligned_vector<std::pair<
            std::string, thin_lens_camera>>     cameras;

    mouse::pos                                  mouse_pos;

    texture_format                              tex_format = UV;

    visionaray::frame_counter                   counter;
    double                                      last_frame_time = 0.0;
    gl::bvh_outline_renderer                    outlines;
    gl::debug_callback                          gl_debug_callback;

    bool                                        render_async  = false;
    std::future<void>                           render_future;
    std::mutex                                  display_mutex;

    void build_scene();

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

private:

    void load_camera(std::string filename);
    void clear_frame();
    void render_hud();
    void render_impl();

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
// Traverse the scene graph to construct geometry, materials and BVH instances
//

struct build_scene_visitor : sg::node_visitor
{
    using node_visitor::apply;

    build_scene_visitor(
            aligned_vector<renderer::host_bvh_type>& bvhs,
            aligned_vector<size_t>& instance_indices,
            aligned_vector<mat4>& instance_transforms,
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
            renderer::bvh_build_strategy build_strategy
            )
        : bvhs_(bvhs)
        , instance_indices_(instance_indices)
        , instance_transforms_(instance_transforms)
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
        , environment_map(nullptr)
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
            environment_map = std::make_shared<visionaray::texture<vec4, 2>>(tex->width(), tex->height());
            environment_map->set_address_mode(tex->get_address_mode());
            environment_map->set_filter_mode(tex->get_filter_mode());
            environment_map->reset(tex->data());
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

        instance_indices_.push_back(~tm.flags());
        instance_transforms_.push_back(current_transform_);

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

        instance_indices_.push_back(~itm.flags());
        instance_transforms_.push_back(current_transform_);

        node_visitor::apply(itm);
    }

    // List of surface properties to derive geom_ids from
    std::vector<std::pair<std::shared_ptr<sg::material>, std::shared_ptr<sg::texture>>> surfaces;

    // Current transform along the path
    mat4 current_transform_ = mat4::identity();


    // Storage bvhs
    aligned_vector<renderer::host_bvh_type>& bvhs_;

    // Indices to construct instances from
    aligned_vector<size_t>& instance_indices_;

    // Transforms to construct instances from
    aligned_vector<mat4>& instance_transforms_;

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
    std::shared_ptr<visionaray::texture<vec4, 2>> environment_map;

    // Assign consecutive prim ids
    unsigned current_prim_id_ = 0;

    // Assign consecutive geom ids for each encountered material
    unsigned current_geom_id_ = 0;

    // Index into the bvh list
    unsigned current_bvh_index_ = 0;

    // Index into the instance list
    unsigned current_instance_index_ = 0;

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
        host_bvhs.resize(1);
        if (build_strategy == LBVH)
        {
            lbvh_builder builder;

            host_bvhs[0] = builder.build(host_bvh_type{}, mod.primitives.data(), mod.primitives.size());
        }
        else
        {
            binned_sah_builder builder;
            builder.enable_spatial_splits(build_strategy == Split);

            host_bvhs[0] = builder.build(host_bvh_type{}, mod.primitives.data(), mod.primitives.size());
        }
    }
    else
    {
        reset_flags_visitor reset_visitor;
        mod.scene_graph->accept(reset_visitor);

        aligned_vector<size_t> instance_indices;
        aligned_vector<mat4> instance_transforms;

        build_scene_visitor build_visitor(
                host_bvhs,
                instance_indices,
                instance_transforms,
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
                build_strategy
                );
        mod.scene_graph->accept(build_visitor);

        host_instances.resize(instance_indices.size());
        for (size_t i = 0; i < instance_indices.size(); ++i)
        {
            size_t index = instance_indices[i];
            host_instances[i] = host_bvhs[index].inst(instance_transforms[i]);
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
                newmat.cd = disney->base_color.xyz();
                newmat.cs = vec3(0.0f);
                newmat.ior = vec3(disney->ior);
                newmat.transmission = disney->refractive; // TODO
                if (newmat.transmission > 0.0f)
                {
                    newmat.illum = 4;
                    newmat.cs = vec3(disney->spec_trans);
                }
            }
            else if (auto obj = std::dynamic_pointer_cast<sg::obj_material>(surf.first))
            {
                // TODO: consolidate model::material and obj_material!
                newmat.ca = obj->ca;
                newmat.cd = obj->cd;
                newmat.cs = obj->cs;
                newmat.ce = obj->ce;
                newmat.cr = obj->cr;
                newmat.ior = obj->ior;
                newmat.absorption = obj->absorption;
                newmat.transmission = obj->transmission;
                newmat.specular_exp = obj->specular_exp;
                newmat.illum = obj->illum;
            }
            else if (auto glass = std::dynamic_pointer_cast<sg::glass_material>(surf.first))
            {
                newmat.transmission = 1.0f; // could be anything > 0.0f
                // This is how model material is later transformed to glass
                newmat.cd = glass->ct;
                newmat.cs = glass->cr;
                newmat.ior = glass->ior;
                newmat.illum = 4;
            }
            mod.materials.emplace_back(newmat); // TODO

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

        environment_map = build_visitor.environment_map;

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
        rt.clear_color_buffer();
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
    auto rgba = color[(h - 1 - y) * w + x];

    int num_nodes = 0;
    int num_leaves = 0;

    float focal_dist = cam.get_focal_distance();
    float lens_radius = cam.get_lens_radius();

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


    static const std::string camera_file_base = "visionaray-camera";
    static const std::string camera_file_suffix = ".txt";

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

    vec3 amb;

    std::vector<std::string> algo_names;
    algo_names.push_back("Simple");
    algo_names.push_back("Whitted");
    algo_names.push_back("Path Tracing");

    std::vector<std::string> ssaa_modes;
    ssaa_modes.push_back("1x SSAA");
    ssaa_modes.push_back("2x SSAA");
    ssaa_modes.push_back("4x SSAA");
    ssaa_modes.push_back("8x SSAA");
    if (ImGui::BeginTabBar("Menu"))
    {
        // Overview tab
        if (ImGui::BeginTabItem("Overview"))
        {
            ImGui::Spacing();
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
                ImGui::Text("SPP: %7u", std::max(1U, frame_num));
            }
            else
            {
                ImGui::Text("SPP: %dx SSAA", ssaa_samples);
            }
            ImGui::SameLine();
            ImGui::Spacing();
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
                    if (host_top_level_bvh.num_nodes() > 0)
                    {
                        outlines.init(host_top_level_bvh);
                    }
                    else
                    {
                        outlines.init(host_bvhs[0]);
                    }
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
            ImGui::Text("Device: %s", rt.mode() == host_device_rt::GPU ? "GPU" : "CPU");
            int nbit = sizeof(ssaa_samples) * 8;
            int ssaa_log2 = nbit - 1 - detail::clz(ssaa_samples);
            std::string currentssaa = ssaa_modes[ssaa_log2];
            if (ImGui::BeginCombo("SPP", currentssaa.c_str()))
            {
                for (size_t i = 0; i < ssaa_modes.size(); ++i)
                {
                    bool selectedssaa = currentssaa == ssaa_modes[i];

                    if (ImGui::Selectable(ssaa_modes[i].c_str(), selectedssaa))
                    {
                        currentssaa = ssaa_modes[i].c_str();
                        if (ssaa_modes[i] == ssaa_modes[0])
                        {
                            ssaa_samples = 1;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if (ssaa_modes[i] == ssaa_modes[1])
                        {
                            ssaa_samples = 2;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if (ssaa_modes[i] == ssaa_modes[2])
                        {
                            ssaa_samples = 4;
                            if (algo != Pathtracing)
                            {
                                counter.reset();
                                clear_frame();
                            }
                        }
                        else if ( ssaa_modes[i] == ssaa_modes[3])
                        {

                            ssaa_samples = 8;
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
            amb = ambient.x < 0.0f ? vec3(0.0f) : ambient;
            if (ImGui::ColorEdit4("Ambient Color", amb.data(), ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float))
            {
                ambient = amb;
                clear_frame();
            }

            ImGui::Spacing();
            ImGui::Spacing();
            std::string currentalgo = "";
            if (algo == Simple)
            {
                currentalgo = algo_names[0];
            }
            else if (algo == Whitted)
            {
                currentalgo = algo_names[1];
            }
            else if (algo == Pathtracing)
            {
                currentalgo = algo_names[2];
            }

            if (ImGui::BeginCombo("Algorithm", currentalgo.c_str()))
            {
                for (size_t i = 0; i < algo_names.size(); ++i)
                {
                    bool selected = currentalgo == algo_names[i];

                    if (ImGui::Selectable(algo_names[i].c_str(), selected))
                    {
                        currentalgo = algo_names[i].c_str();
                        if (algo_names[i] == algo_names[0])
                        {
                            rt.set_double_buffering(true);
                            algo = Simple;
                            counter.reset();
                            clear_frame();
                        }
                        else if (algo_names[i] == algo_names[1])
                        {
                            rt.set_double_buffering(true);
                            algo = Whitted;
                            counter.reset();
                            clear_frame();
                        }
                        else if (algo_names[i] == algo_names[2])
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

            if (bc == 1){
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

    camera_t camx;
    if (use_dof && algo == Pathtracing)
    {
        camx = cam;
    }
    else
    {
        camx = static_cast<pinhole_camera>(cam);
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
                        ssaa_samples
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
                        ssaa_samples
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
                    ssaa_samples
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
                    camx,
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
                    camx,
                    frame_num,
                    algo,
                    ssaa_samples
                    );
        }
    }
#endif

    last_frame_time = counter.register_frame();

#if VSNRAY_COMMON_HAVE_PTEX
//  if (ptex_textures.size() > 0)
//  {
//      PtexCache::Stats stats;
//      ptex_textures[0].cache.get()->get()->getStats(stats);
//      std::cout << "Mem used:        " << stats.memUsed << '\n';
//      std::cout << "Peak mem used:   " << stats.peakMemUsed << '\n';
//      std::cout << "Files open:      " << stats.filesOpen << '\n';
//      std::cout << "Peak files open: " << stats.peakFilesOpen << '\n';
//      std::cout << "Files accessed:  " << stats.filesAccessed << '\n';
//      std::cout << "File reopens:    " << stats.fileReopens << '\n';
//      std::cout << "Block reads:     " << stats.blockReads << '\n';
//  }
#endif

    if (use_headlight)
    {
        point_lights.erase(point_lights.end() - 1);
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
    static const std::string camera_file_base = "visionaray-camera";
    static const std::string camera_file_suffix = ".txt";

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

    case 'b':
        show_bvh = !show_bvh;

        if (show_bvh)
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

    rend.build_scene();

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

#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        rend.device_bvh = renderer::device_bvh_type(rend.host_bvhs[0]);
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
