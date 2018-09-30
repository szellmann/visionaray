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
#include <visionaray/thin_lens_camera.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/make_materials.h>
#include <common/model.h>
#include <common/sg.h>
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

        add_cmdline_option( cl::makeOption<std::set<std::string>&>(
            cl::Parser<>(),
            "filenames",
            cl::Desc("Input files in wavefront obj format"),
            cl::Positional,
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
    bool                                        use_headlight   = true;
    bool                                        use_dof         = false;
    bool                                        show_hud        = true;
    bool                                        show_hud_ext    = true;
    bool                                        show_bvh        = false;


    std::set<std::string>                       filenames;
    std::string                                 initial_camera;

    model                                       mod;
    vec3                                        ambient         = vec3(-1.0f);

    index_bvh<host_bvh_type::bvh_inst>          host_top_level_bvh;
    aligned_vector<host_bvh_type>               host_bvhs;
    aligned_vector<host_bvh_type::bvh_inst>     host_instances;
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
    thin_lens_camera                            cam;

    mouse::pos                                  mouse_pos;

    visionaray::frame_counter                   counter;
    gl::bvh_outline_renderer                    outlines;
    gl::debug_callback                          gl_debug_callback;

    bool                                        render_async  = false;
    std::future<void>                           render_future;
    std::mutex                                  display_mutex;

    void build_bvhs();

protected:

    void on_close();
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

private:

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
};


//-------------------------------------------------------------------------------------------------
// Traverse the scene graph to construct geometry, materials and BVH instances
//

struct build_bvhs_visitor : sg::node_visitor
{
    using node_visitor::apply;

    build_bvhs_visitor(
            aligned_vector<renderer::host_bvh_type>& bvhs,
            aligned_vector<size_t>& instance_indices,
            aligned_vector<mat4>& instance_transforms,
            aligned_vector<vec3>& shading_normals,
            aligned_vector<vec3>& geometric_normals,
            aligned_vector<vec2>& tex_coords
#if VSNRAY_COMMON_HAVE_PTEX
          , aligned_vector<ptex::face_id_t>& face_ids
#endif
            )
        : bvhs_(bvhs)
        , instance_indices_(instance_indices)
        , instance_transforms_(instance_transforms)
        , shading_normals_(shading_normals)
        , geometric_normals_(geometric_normals)
        , tex_coords_(tex_coords)
#if VSNRAY_COMMON_HAVE_PTEX
        , face_ids_(face_ids)
#endif
    {
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

        if (sp.flags() == 0 && sp.material() && sp.textures().size() > 0)
        {
            std::shared_ptr<sg::material> material = sp.material();
            std::shared_ptr<sg::texture> texture = sp.textures()[0];

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
            bvhs_.emplace_back(build<renderer::host_bvh_type>(
                    triangles.data(),
                    triangles.size(),
                    false//builder == Split
                    ));

            tm.flags() = ~(bvhs_.size() - 1);
        }

        instance_indices_.push_back(~tm.flags());
        instance_transforms_.push_back(current_transform_);

        node_visitor::apply(tm);
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

#if VSNRAY_COMMON_HAVE_PTEX
    aligned_vector<ptex::face_id_t>& face_ids_;
#endif

    // Assign consecutive prim ids
    unsigned current_prim_id_ = 0;

    // Assign consecutive geom ids for each encountered material
    unsigned current_geom_id_ = 0;

    // Index into the bvh list
    unsigned current_bvh_index_ = 0;

    // Index into the instance list
    unsigned current_instance_index_ = 0;

};


//-------------------------------------------------------------------------------------------------
// Build bvhs
//

void renderer::build_bvhs()
{
//  timer t;

    std::cout << "Creating BVH...\n";

    if (mod.scene_graph == nullptr)
    {
        // Single BVH
        host_bvhs.resize(1);
        host_bvhs[0] = build<host_bvh_type>(
                mod.primitives.data(),
                mod.primitives.size(),
                builder == Split
                );
    }
    else
    {
        reset_flags_visitor reset_visitor;
        mod.scene_graph->accept(reset_visitor);

        aligned_vector<size_t> instance_indices;
        aligned_vector<mat4> instance_transforms;

        build_bvhs_visitor build_visitor(
                host_bvhs,
                instance_indices,
                instance_transforms,
                mod.shading_normals, // TODO!!!
                mod.geometric_normals,
                mod.tex_coords
#if VSNRAY_COMMON_HAVE_PTEX
              , mod.ptex_tex_coords
#endif
                );
        mod.scene_graph->accept(build_visitor);

        host_instances.resize(instance_indices.size());
        for (size_t i = 0; i < instance_indices.size(); ++i)
        {
            size_t index = instance_indices[i];
            host_instances[i] = host_bvhs[index].inst(instance_transforms[i]);
        }

        host_top_level_bvh = build<index_bvh<host_bvh_type::bvh_inst>>(
                host_instances.data(),
                host_instances.size(),
                false
                );


        mod.tex_format = model::UV;

#if VSNRAY_COMMON_HAVE_PTEX
        // Simply check the first texture of the first surface
        // Scene has either Ptex textures, or it doesn't
        if (build_visitor.surfaces.size() > 0
            && std::dynamic_pointer_cast<sg::ptex_texture>(build_visitor.surfaces[0].second) != nullptr)
        {
            mod.tex_format = model::Ptex;
            mod.ptex_textures.resize(build_visitor.surfaces.size());
        }
#endif

        if (mod.tex_format == model::UV)
        {
            mod.textures.resize(build_visitor.surfaces.size());
        }

        for (size_t i = 0; i < build_visitor.surfaces.size(); ++i)
        {
            auto const& surf = build_visitor.surfaces[i];

            auto disney = std::dynamic_pointer_cast<sg::disney_material>(surf.first);
            assert(disney != nullptr);

            model::material_type newmat = {};
            newmat.cd = disney->base_color.xyz();
            newmat.cs = vec3(0.0f);
            newmat.ior = vec3(disney->ior);
            newmat.transmission = disney->refractive; // TODO
            if (newmat.transmission > 0.0f)
            {
                newmat.illum = 4;
                newmat.cs = vec3(disney->spec_trans);
            }
            mod.materials.emplace_back(newmat); // TODO

#if VSNRAY_COMMON_HAVE_PTEX
            auto ptex_tex = std::dynamic_pointer_cast<sg::ptex_texture>(surf.second);
            if (ptex_tex != nullptr)
            {
                mod.ptex_textures[i] = { ptex_tex->filename(), ptex_tex->cache() };
            }
#else
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
#endif
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

    ImGui::Text("FPS: %6.2f", counter.register_frame());
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
    ImGui::SameLine();
    ImGui::Text("Device: %s", rt.mode() == host_device_rt::GPU ? "GPU" : "CPU");

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

    vec3 amb = ambient.x < 0.0f ? vec3(0.0f) : ambient;
    if (ImGui::InputFloat3("Ambient Intensity", amb.data()))
    {
        ambient = amb;
        clear_frame();
    }

    if (ImGui::Checkbox("DoF", &use_dof) && algo == Pathtracing)
    {
        clear_frame();

        if (algo != Pathtracing)
        {
            std::cerr << "Warning: setting only affects pathtracing algorithm\n";
        }
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(200);
    if (ImGui::SliderFloat("", &focal_dist, 0.1, 100.0f, "Focal Dist. %.1f"))
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
    ImGui::SameLine();
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

    ImGui::End();
}

void renderer::render_impl()
{
    point_lights.clear();

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

            if (mod.tex_format == model::UV)
            {
                render_instances_cpp(
                        host_top_level_bvh,
                        mod.geometric_normals,
                        mod.shading_normals,
                        mod.tex_coords,
                        generic_materials,
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
            else if (mod.tex_format == model::Ptex)
            {
                render_instances_ptex_cpp(
                        host_top_level_bvh,
                        mod.geometric_normals,
                        mod.shading_normals,
                        mod.ptex_tex_coords,
                        generic_materials,
                        mod.ptex_textures,
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

#if VSNRAY_COMMON_HAVE_PTEX
//  if (mod.ptex_textures.size() > 0)
//  {
//      PtexCache::Stats stats;
//      mod.ptex_textures[0].cache.get()->get()->getStats(stats);
//      std::cout << "Mem used:        " << stats.memUsed << '\n';
//      std::cout << "Peak mem used:   " << stats.peakMemUsed << '\n';
//      std::cout << "Files open:      " << stats.filesOpen << '\n';
//      std::cout << "Peak files open: " << stats.peakFilesOpen << '\n';
//      std::cout << "Files accessed:  " << stats.filesAccessed << '\n';
//      std::cout << "File reopens:    " << stats.fileReopens << '\n';
//      std::cout << "Block reads:     " << stats.blockReads << '\n';
//  }
#endif
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
    static const std::string camera_filename = "visionaray-camera.txt";

    switch (event.key())
    {
    case '1':
        std::cout << "Switching algorithm: simple\n";
        rt.set_double_buffering(true);
        algo = Simple;
        counter.reset();
        clear_frame();
        break;

    case '2':
        std::cout << "Switching algorithm: whitted\n";
        rt.set_double_buffering(true);
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
    if (render_future.valid() && algo != Pathtracing)
    {
        render_future.wait();
    }

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

	if (rend.filenames.empty())
	{
		std::cout << rend.cmd_line_inst().help(argv[0]) << "\n";
		return EXIT_FAILURE;
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

    rend.build_bvhs();

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
