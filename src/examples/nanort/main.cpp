// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <memory>

#include <GL/glew.h>

#include <nanort.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/get_normal.h>
#include <visionaray/random_sampler.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

#include <common/model.h>
#include <common/obj_loader.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<float>;

    using bvh_type = nanort::BVHAccel<
            float,
            nanort::TriangleMesh<float>,
            nanort::TriangleSAHPred<float>,
            nanort::TriangleIntersector<>
            >;

    renderer()
        : viewer_type(512, 512, "Visionaray NanoRT Example")
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
    }

    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    std::string                                 filename;

    model mod;
    aligned_vector<float>                       vertices;
    aligned_vector<unsigned>                    faces;
    bvh_type                                    nano_bvh;
    unsigned                                    frame_num       = 0;

protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function, implements the AO kernel
//

void renderer::on_display()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = R::scalar_type;
    using C = vector<4, S>;
    using V = vector<3, S>;

    auto sparams = make_sched_params(
            pixel_sampler::jittered_blend_type{},
            cam,
            host_rt
            );


    auto bgcolor = background_color();

    timer t;

    host_sched.frame([&](R ray, random_sampler<S>& samp) -> result_record<S>
    {
        result_record<S> result;
        result.color = C(bgcolor, 1.0f);

        // Closest hit
        nanort::Ray<float> nano_ray;
        nano_ray.min_t = 0.0f;
        nano_ray.max_t = FLT_MAX;
        nano_ray.org[0] = ray.ori.x;
        nano_ray.org[1] = ray.ori.y;
        nano_ray.org[2] = ray.ori.z;
        nano_ray.dir[0] = ray.dir.x;
        nano_ray.dir[1] = ray.dir.y;
        nano_ray.dir[2] = ray.dir.z;

        nanort::TriangleIntersector<> nano_intersector(
                vertices.data(),
                faces.data(),
                sizeof(float) * 3
                );

        result.hit = nano_bvh.Traverse(nano_ray, nanort::BVHTraceOptions(), nano_intersector);

        // AO
        if (result.hit)
        {
            hit_record<R, primitive<unsigned>> hit_rec;
            hit_rec.hit = result.hit;
            hit_rec.t = nano_intersector.intersection.t;
            hit_rec.u = nano_intersector.intersection.u;
            hit_rec.v = nano_intersector.intersection.v;
            hit_rec.prim_id = nano_intersector.intersection.prim_id;
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            C clr(1.0);

            auto n = get_normal(
                mod.geometric_normals.data(),
                hit_rec,
                basic_triangle<3, float>{},
                normals_per_face_binding{}
                );

            // Make an ortho basis (TODO: move to library)
            auto w = n;
            auto v = select(
                    abs(w.x) > abs(w.y),
                    normalize( V(-w.z, S(0.0), w.x) ),
                    normalize( V(S(0.0), w.z, -w.y) )
                    );
            auto u = cross(v, w);

            static const int AO_Samples = 8;
            S radius(0.1);


            for (int i = 0; i < AO_Samples; ++i)
            {
                auto sp = cosine_sample_hemisphere(samp.next(), samp.next());

                auto dir = normalize( sp.x * u + sp.y * v + sp.z * w );

                R ao_ray;
                ao_ray.ori = hit_rec.isect_pos + dir * S(1E-3f);
                ao_ray.dir = dir;

                nanort::Ray<float> nano_ao_ray;
                nano_ao_ray.min_t = 0.0f;
                nano_ao_ray.max_t = radius;
                nano_ao_ray.org[0] = ao_ray.ori.x;
                nano_ao_ray.org[1] = ao_ray.ori.y;
                nano_ao_ray.org[2] = ao_ray.ori.z;
                nano_ao_ray.dir[0] = ao_ray.dir.x;
                nano_ao_ray.dir[1] = ao_ray.dir.y;
                nano_ao_ray.dir[2] = ao_ray.dir.z;

                bool hit = nano_bvh.Traverse(nano_ao_ray, nanort::BVHTraceOptions(), nano_intersector);
                clr = select(
                        hit,
                        clr - S(1.0f / AO_Samples),
                        clr
                        );
            }

            result.color = select( result.hit, C(clr.xyz(), S(1.0)), result.color );
        }

        return result;
    }, sparams, ++frame_num);


    // display the rendered image

    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();

    std::cout << "Time to frame: " << t.elapsed() << '\n';
}


//-------------------------------------------------------------------------------------------------
// mouse handling
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.get_buttons() != mouse::NoButton)
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
        std::cerr << "Failed loading obj model: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    // Bring triangles into nanort format..
    rend.vertices.resize(rend.mod.primitives.size() * 9);
    rend.faces.resize(rend.mod.primitives.size() * 3);
    for (size_t i = 0; i < rend.mod.primitives.size(); ++i)
    {
        auto t = rend.mod.primitives[i];
        auto v1 = t.v1;
        auto v2 = t.v1 + t.e1;
        auto v3 = t.v1 + t.e2;
        memcpy(rend.vertices.data() + i * 9,     v1.data(), sizeof(float) * 3);
        memcpy(rend.vertices.data() + i * 9 + 3, v2.data(), sizeof(float) * 3);
        memcpy(rend.vertices.data() + i * 9 + 6, v3.data(), sizeof(float) * 3);
        rend.faces[i * 3]     = i * 3;
        rend.faces[i * 3 + 1] = i * 3 + 1;
        rend.faces[i * 3 + 2] = i * 3 + 2;
    }

    std::cout << "Creating BVH...\n";
    timer t;

    nanort::TriangleMesh<float> nano_mesh(rend.vertices.data(), rend.faces.data(), sizeof(float) * 3);
    nanort::TriangleSAHPred<float> nano_pred(rend.vertices.data(), rend.faces.data(), sizeof(float) * 3);
    rend.nano_bvh.Build(rend.faces.size() / 3, nanort::BVHBuildOptions<float>(), nano_mesh, nano_pred);

    std::cout << "Ready\n";
    std::cout << "Time to build: " << t.elapsed() << '\n';
    std::cout << "###############\n";

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
