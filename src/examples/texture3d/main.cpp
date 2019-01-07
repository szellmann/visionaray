// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include <GL/glew.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

#include <common/make_materials.h>
#include <common/model.h>
#include <common/obj_loader.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float4>;
    using material_type = plastic<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray 3D Texture Example")
        , host_sched(8)
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

        add_cmdline_option( cl::makeOption<vec3i&, cl::ScalarType>(
            [&](StringRef name, StringRef /*arg*/, vec3i& value)
            {
                cl::Parser<>()(name + "-w", cmd_line_inst().bump(), value.x);
                cl::Parser<>()(name + "-h", cmd_line_inst().bump(), value.y);
                cl::Parser<>()(name + "-d", cmd_line_inst().bump(), value.z);
            },
            "texsize",
            cl::Desc("Size of the 3D texture"),
            cl::ArgDisallowed,
            cl::init(this->texsize)
            ) );

        add_cmdline_option( cl::makeOption<vec3&, cl::ScalarType>(
            [&](StringRef name, StringRef /*arg*/, vec3& value)
            {
                cl::Parser<>()(name + "-r", cmd_line_inst().bump(), value.x);
                cl::Parser<>()(name + "-g", cmd_line_inst().bump(), value.y);
                cl::Parser<>()(name + "-b", cmd_line_inst().bump(), value.z);
            },
            "color1",
            cl::Desc("First color"),
            cl::ArgDisallowed,
            cl::init(this->color1)
            ) );

        add_cmdline_option( cl::makeOption<vec3&, cl::ScalarType>(
            [&](StringRef name, StringRef /*arg*/, vec3& value)
            {
                cl::Parser<>()(name + "-r", cmd_line_inst().bump(), value.x);
                cl::Parser<>()(name + "-g", cmd_line_inst().bump(), value.y);
                cl::Parser<>()(name + "-b", cmd_line_inst().bump(), value.z);
            },
            "color2",
            cl::Desc("Second color"),
            cl::ArgDisallowed,
            cl::init(this->color2)
            ) );
    }

    void init_texture()
    {
        // Build up a 3D texture that stores RGBA colors.
        tex = texture<vec4, 3>(texsize.x, texsize.y, texsize.z);

        // Construct the 3D texture in an array.
        aligned_vector<vec4> arr(tex.width() * tex.height() * tex.depth());

        for (size_t z = 0; z < tex.depth(); ++z)
        {
            for (size_t y = 0; y < tex.height(); ++y)
            {
                for (size_t x = 0; x < tex.width(); ++x)
                {
                    size_t index = z * tex.width() * tex.height() + y * tex.width() + x;

                    int col_index = x % 2 == y % 2;
                    col_index = z % 2 == 0 ? col_index : !col_index;

                    arr[index] = col_index == 0
                                        ? vec4(color1, 1.0f)
                                        : vec4(color2, 1.0f)
                                        ;
                }
            }
        }

        // Initialize the 3D texture from the array.
        tex.reset(arr.data());
        // Nearest neighbor filtering. Alternatives are:
        //  Linear, BSpline, CardinalSpline
        tex.set_filter_mode(Nearest);
        // Clamp border texels. Could alternatively use:
        //  Wrap, Mirror
        tex.set_address_mode(Clamp);


        // Build up some simple 3D texture coordinates
        for (auto tri : mod.primitives)
        {
            auto v1 = tri.v1;
            auto v2 = tri.e1 + tri.v1;
            auto v3 = tri.e2 + tri.v1;

            tex_coords.push_back((v1 - mod.bbox.min) / mod.bbox.size());
            tex_coords.push_back((v2 - mod.bbox.min) / mod.bbox.size());
            tex_coords.push_back((v3 - mod.bbox.min) / mod.bbox.size());
        }
    }

    pinhole_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    std::string                                 filename;

    model mod;
    bvh<model::triangle_type>                   host_bvh;
    unsigned                                    frame_num       = 0;
    vec3                                        ambient         = vec3(1.0f, 1.0f, 1.0f);

    texture<vec4, 3>                            tex;
    aligned_vector<vec3>                        tex_coords;
    aligned_vector<material_type>               materials;
    vec3i                                       texsize         = vec3i(8, 8, 8);
    vec3                                        color1          = vec3(1.0f, 1.0f, 0.7f);
    vec3                                        color2          = vec3(0.4f, 0.0f, 0.0f);

protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    // number of bounces used during path tracing.
    auto bounces    = 4U;
    // eps is used when generating secondary rays.
    // to avoid self-intersection.
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );

    //-----------------------------------------------------
    // Setup scheduler parameters.
    //

    // We use path tracing. We therefore use the
    // jittered_blend_type pixel sampler. That way
    // a rand generator is created, and noisy
    // images are blended later on.

    // Note how we set sfactor and dfactor based on
    // frame_num so that pixel blending works.

    // Note: for the jittered_blend_type pixel sampler to
    // work properly, you have to count frames yourself!
    // (check the occurrences of renderer::frame_num in
    // this file!)

    pixel_sampler::jittered_blend_type blend_params;
    float alpha = 1.0f / ++frame_num;
    blend_params.sfactor = alpha;
    blend_params.dfactor = 1.0f - alpha;

    // Note: alternative samplers are:
    //  pixel_sampler::uniform_type
    //  pixel_sampler::ssaa_type<[1|2|4|8}>
    // (uniform_type and ssaa_type<1> will behave the same!)
    // You can also leave this argument out, then you'll get
    // the default: pixel_sampler::uniform_type
    auto sparams = make_sched_params(
            blend_params,
            cam,          // the camera object (note: could also be two matrices!)
            host_rt       // render target, that's where we store the pixel result.
            );

    //-----------------------------------------------------
    // Setup kernel parameters
    //

    // The default kernels use a dedicated kernel parameter
    // type (custom kernels may obtain state in any other way,
    // e.g. by direct lambda capture).

    // *_ref types have the same interface like the
    // types they refer to, but internally only store
    // light weight accessors. *_ref's come in handy e.g.
    // if you've already created a BVH or a texture and
    // have copied it to the GPU already, and now only
    // want to refer to it in your code, w/o doing the
    // copying again.

    using bvh_ref = bvh<model::triangle_type>::bvh_ref;
    using tex_ref = texture_ref<vec4, 3>;

    // Algorithms like closest_hit(), which are called
    // by the path tracing kernel, perform range-based
    // traversal over a set of primitives. In Visionaray,
    // BVHs are also primitives, so we need to construct
    // a list of BVHs to make range-based traversal work.
    aligned_vector<bvh_ref> primitives;
    primitives.push_back(host_bvh.ref());

    // The same is true for textures.
    // aligned_vector<> is a std::vector<> with a custom
    // allocator. Use it in conjunction w/ SIMD traversal
    // where there are restrictions on alignment properties.
    aligned_vector<tex_ref> textures;
    for (size_t i = 0; i < materials.size(); ++i)
    {
        textures.emplace_back(tex_ref(tex));
    }

    // make_kernel_params needs (!) lights
    // TODO: fix this in visionaray API!
    struct no_lights {};
    no_lights* ignore = 0;

    // Construct a parameter object that is
    // compatible with the builtin path tracing kernel.
    auto kparams = make_kernel_params(
            normals_per_vertex_binding{},
            primitives.data(),
            primitives.data() + primitives.size(),
            mod.geometric_normals.data(),
            mod.shading_normals.data(),
            tex_coords.data(),
            materials.data(),
            textures.data(),
            ignore,
            ignore,
            bounces,
            epsilon,
            vec4(background_color(), 1.0f),
            vec4(ambient, 1.0f)
            );

    //-----------------------------------------------------
    // Naive path tracing with the builtin kernel.
    //

    // Instantiate the path tracing kernel, and
    // call it by executing the scheduler's
    // frame() function.
    pathtracing::kernel<decltype(kparams)> kernel;
    kernel.params = kparams;
    host_sched.frame(kernel, sparams);


    // Display the rendered image with OpenGL.

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // You could also directly access host_rt::color()
    // or host_rt::depth() (this render target however
    // doesn't store a depth buffer).
    host_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// mouse handling
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.buttons() != mouse::NoButton)
    {
        frame_num = 0;
        host_rt.clear_color_buffer();
    }

    viewer_type::on_mouse_move(event);
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    frame_num = 0;
    host_rt.clear_color_buffer();

    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    try
    {
        visionaray::load_obj(rend.filename, rend.mod);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Convert generic materials to viewer's material type
    rend.materials = make_materials(
            renderer::material_type{},
            rend.mod.materials
            );

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    binned_sah_builder builder;

    rend.host_bvh = builder.build<bvh<model::triangle_type>>(
            rend.mod.primitives.data(),
            rend.mod.primitives.size()
            );

    std::cout << "Creating 3D texture and texture coordinates...\n";

    rend.init_texture();

    std::cout << "Ready\n";

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    rend.cam.view_all( rend.mod.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
