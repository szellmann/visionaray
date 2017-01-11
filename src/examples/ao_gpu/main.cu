// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include <GL/glew.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/get_normal.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#include <visionaray/random_sampler.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

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
    using host_ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray Ambient Occlusion GPU Example")
        , device_sched(8, 8)
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

        add_cmdline_option( cl::makeOption<bvh_build_strategy&>({
                { "default",            Binned,         "Binned SAH" },
                { "split",              Split,          "Binned SAH with spatial splits" }
            },
            "bvh",
            cl::Desc("BVH build strategy"),
            cl::ArgRequired,
            cl::init(this->builder)
            ) );

        add_cmdline_option( cl::makeOption<int&>(
            cl::Parser<>(),
            "samples",
            cl::Desc("Number of shadow rays for ambient occlusion"),
            cl::ArgRequired,
            cl::init(this->AO_Samples)
            ) );

        add_cmdline_option( cl::makeOption<float&>(
            cl::Parser<>(),
            "radius",
            cl::Desc("Ambient occlusion radius"),
            cl::ArgRequired,
            cl::init(this->AO_Radius)
            ) );
    }

    enum bvh_build_strategy
    {
        Binned = 0,  // Binned SAH builder, no spatial splits
        Split        // Split BVH, also binned and with SAH
    };

    camera                                              cam;
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>  d_rt;
    cuda_sched<host_ray_type>                           device_sched;
    bvh_build_strategy                                  builder         = Binned;

    std::string                                         filename;
    std::string                                         initial_camera;

    model mod;
    index_bvh<model::triangle_type>                     host_bvh;
    cuda_index_bvh<model::triangle_type>                d_bvh;
    thrust::device_vector<vec3>                         d_geo_normals;
    unsigned                                            frame_num       = 0;

    int                                                 AO_Samples      = 8;
    float                                               AO_Radius       = 0.1f;

protected:

    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

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

template <typename PrimIt, typename Normals, typename Color>
struct ao_kernel
{
    PrimIt  prims_begin;
    PrimIt  prims_end;
    Normals geometric_normals;
    Color   bgcolor;
    float   AO_Radius;
    int     AO_Samples;


    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(R ray, random_sampler<S>& samp) -> result_record<S>
    {
        using C = vector<4, S>;
        using V = vector<3, S>;

        result_record<S> result;
        result.color = C(bgcolor, 1.0f);

        auto hit_rec = closest_hit(
                ray,
                prims_begin,
                prims_end
                );

        result.hit = hit_rec.hit;

        if (visionaray::any(hit_rec.hit))
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;
            result.isect_pos  = hit_rec.isect_pos;

            C clr(1.0);

            auto n = get_normal(
                geometric_normals,
                hit_rec,
                cuda_index_bvh<model::triangle_type>{},
                normals_per_face_binding{}
                );

            V u;
            V v;
            V w = n;
            make_orthonormal_basis(u, v, w);

            S radius = AO_Radius;

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
                        radius
                        );

                clr = select(
                        ao_rec.hit,
                        clr - S(1.0f / AO_Samples),
                        clr
                        );
            }

            result.color = select( hit_rec.hit, C(clr.xyz(), S(1.0)), result.color );

        }

        return result;
    }
};

template <typename PrimIt, typename Normals, typename Color>
auto make_ao_kernel(
        PrimIt  pbegin,
        PrimIt  pend,
        Normals gn,
        Color   bgcolor,
        float   radius,
        int     samples
        )
    -> ao_kernel<PrimIt, Normals, Color>
{
    return { pbegin, pend, gn, bgcolor, radius, samples };
}


//-------------------------------------------------------------------------------------------------
// Display function, implements the AO kernel
//

void renderer::on_display()
{
    // some setup

    auto sparams = make_sched_params(
            pixel_sampler::jittered_blend_type{},
            cam,
            d_rt
            );


    using bvh_ref = cuda_index_bvh<model::triangle_type>::bvh_ref;

    thrust::device_vector<bvh_ref> bvhs;
    bvhs.push_back(d_bvh.ref());

    auto bgcolor = background_color();

    auto kernel = make_ao_kernel(
            thrust::raw_pointer_cast(bvhs.data()),
            thrust::raw_pointer_cast(bvhs.data() + bvhs.size()),
            thrust::raw_pointer_cast(d_geo_normals.data()),
            bgcolor,
            AO_Radius,
            AO_Samples
            );

    device_sched.frame(kernel, sparams, ++frame_num);


    // display the rendered image

    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// mouse handling
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.get_buttons() != mouse::NoButton)
    {
        frame_num = 0;
        d_rt.clear_color_buffer();
    }

    viewer_type::on_mouse_move(event);
}


//-------------------------------------------------------------------------------------------------
// keyboard handling
//

void renderer::on_key_press(key_event const& event)
{
    static const std::string camera_filename = "visionaray-camera.txt";

    switch (event.key())
    {
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
                d_rt.clear_color_buffer();
                std::cout << "Load camera from file: " << camera_filename << '\n';
            }
        }
        break;

    default:
        break;
    }

    viewer_type::on_key_press(event);
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    d_rt.resize(w, h);

    // TODO: it should be possible to clear the rt before it
    // was initially resized - however rt::begin_frame()
    // throws when no graphics resource was mapped yet!
    frame_num = 0;
    d_rt.clear_color_buffer();

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
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    try
    {
        visionaray::load_obj(rend.filename, rend.mod);
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Creating BVH...\n";

    rend.host_bvh = build<index_bvh<model::triangle_type>>(
            rend.mod.primitives.data(),
            rend.mod.primitives.size(),
            rend.builder == renderer::Split
            );

    std::cout << "Ready\n";

    // Copy data to GPU
    try
    {
        rend.d_bvh = cuda_index_bvh<model::triangle_type>(rend.host_bvh);
        rend.d_geo_normals = rend.mod.geometric_normals;
    }
    catch (std::bad_alloc&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        exit(EXIT_FAILURE);
    }

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
