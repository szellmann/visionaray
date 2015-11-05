// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h> // for make_params(...)
#include <visionaray/material.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/image.h>
#include <common/viewer_glut.h>

#include "make_textures.h"
#include "quad.h"

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Quad primitive for convenience
// For simplicities sake composed of two triangles
//

namespace texture_ex
{

// Inherit from primitive to get prim_id and geom_id
class quad : public primitive<unsigned>
{
public:

    // Primitive types must be default constructible
    quad() = default;

    // Better be coplanar..
    quad(vec3 v1, vec3 v2, vec3 v3, vec3 v4)
        : tri1_(v1, v2 - v1, v3 - v1)
        , tri2_(v3, v4 - v3, v1 - v3)
        , transform_inv_(mat4::identity())
    {
        tri1_.prim_id = 0;
        tri2_.prim_id = 1;
    }

    void set_transform(mat4 const& t)
    {
        transform_inv_ = inverse(t);
    }

    template <typename R>
    R transform_ray(R const& ray)
    {
        using S = typename R::scalar_type;

        R inv_ray;
        inv_ray.ori = (matrix<4, 4, S>(transform_inv_) * vector<4, S>(ray.ori, S(1.0))).xyz();
        inv_ray.dir = (matrix<4, 4, S>(transform_inv_) * vector<4, S>(ray.dir, S(0.0))).xyz();
        return inv_ray;
    }

    inline basic_triangle<3, float> const* triangles() const
    {
        return reinterpret_cast<basic_triangle<3, float> const*>(&tri1_);
    }

private:

    basic_triangle<3, float> tri1_;
    basic_triangle<3, float> tri2_;

    mat4 transform_inv_;

};


template <typename R, typename T>
struct hit_record;

template <typename T>
struct hit_record<basic_ray<T>, quad> : visionaray::hit_record<basic_ray<T>, primitive<unsigned>>
{
    using int_type = typename simd::int_type<T>::type;

    int_type tri_id;
};

template <
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline std::array<hit_record<ray, quad>, simd::num_elements<FloatT>::value> unpack(
        hit_record<visionaray::basic_ray<FloatT>, quad> const& hr
        )
{
    using float_array = typename simd::aligned_array<FloatT>::type;
    using int_array = typename simd::aligned_array<typename simd::int_type<FloatT>::type>::type;

    using simd::store;

    int_array hit;
    store(hit, hr.hit.i);

    int_array prim_id;
    store(prim_id, hr.prim_id);

    int_array geom_id;
    store(geom_id, hr.geom_id);

    float_array t;
    store(t, hr.t);

    auto isect_pos = unpack(hr.isect_pos);

    float_array u;
    store(u, hr.u);

    float_array v;
    store(v, hr.v);

    int_array tri_id;
    store(tri_id, hr.tri_id);

    std::array<hit_record<ray, quad>, simd::num_elements<FloatT>::value> result;
    for (size_t i = 0; i < simd::num_elements<FloatT>::value; ++i)
    {
        result[i].hit       = hit[i] != 0;
        result[i].prim_id   = prim_id[i];
        result[i].geom_id   = geom_id[i];
        result[i].isect_pos = isect_pos[i];
        result[i].u         = u[i];
        result[i].v         = v[i];
        result[i].tri_id    = tri_id[i];
    }
    return result;
}

template <typename T, typename Cond>
inline void update_if(
        hit_record<basic_ray<T>, quad>&         dst,
        hit_record<basic_ray<T>, quad> const&   src,
        Cond const&                             cond
        )
{
    dst.hit    |= cond;
    dst.t       = select( cond, src.t, dst.t );
    dst.tri_id  = select( cond, src.tri_id, dst.tri_id );
    dst.prim_id = select( cond, src.prim_id, dst.prim_id );
    dst.geom_id = select( cond, src.geom_id, dst.geom_id );
    dst.u       = select( cond, src.u, dst.u );
    dst.v       = select( cond, src.v, dst.v );
}


} // texture_ex

namespace visionaray
{

// redeclare, this is our own namespace
//template <typename P, typename NB>
//struct num_normals;

template <typename NB>
struct num_normals<texture_ex::quad, NB>
{
    enum { value = 1 };
};
#warning("FIXME: move num_normals to namespace texture_ex")
} // visionaray

namespace texture_ex
{


// Custom intersect method

template <typename T>
inline hit_record<basic_ray<T>, quad> intersect(
        basic_ray<T> const& r,
        quad                q
        )
{
    auto triangles = q.triangles();

    auto hr = closest_hit( q.transform_ray(r), triangles, triangles + 2 );

    hit_record<basic_ray<T>, quad> result;

    result.hit      = hr.hit;
    result.t        = hr.t;
    result.tri_id   = hr.prim_id;
    result.u        = hr.u;
    result.v        = hr.v;

    result.prim_id  = q.prim_id;
    result.geom_id  = q.geom_id;

    return result;
}


// We use textures - need to implement get_tex_coord()

template <typename TexCoords, typename R>
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords                                   tex_coords,
        hit_record<R, quad> const&                  hr,
        quad                                        /* */
        )
    -> typename std::iterator_traits<TexCoords>::value_type
{
    return select( hr.tri_id == 0, 
            lerp(
                tex_coords[hr.prim_id * 4    ],
                tex_coords[hr.prim_id * 4 + 1],
                tex_coords[hr.prim_id * 4 + 2],
                hr.u,
                hr.v
                ),
            lerp(
                tex_coords[hr.prim_id * 4 + 2],
                tex_coords[hr.prim_id * 4 + 3],
                tex_coords[hr.prim_id * 4 + 0],
                hr.u,
                hr.v
                )
            );
}

} // texture_ex


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float4>;
    using texture_type  = texture<vector<3, unorm<8>>, NormalizedFloat, 2>;

    renderer()
        : viewer_type(1024, 512, "Visionaray Texture Example")
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
    {
        make_frame();
    }

    void make_frame()
    {
        primitives.clear();
        materials.clear();
        textures.clear();
        normals.clear();
        tex_coords.clear();

        // floor quad
        primitives.emplace_back(
                vec3(-6.0f, -0.82f, -3.0f),
                vec3( 6.0f, -0.82f, -3.0f),
                vec3( 6.0f, -0.82f,  3.0f),
                vec3(-6.0f, -0.82f,  3.0f)
                );


        // textured quads
        for (int i = 0; i < num_quads; ++i)
        {
            if (i >= 0 && i < 12)
            {
                primitives.emplace_back(
                        vec3(-0.8f, -0.8f,  0.3f),
                        vec3( 0.8f, -0.8f,  0.3f),
                        vec3( 0.8f,  0.8f,  0.3f),
                        vec3(-0.8f,  0.8f,  0.3f)
                        );
            }
            else
            {
                primitives.emplace_back(
                        vec3(-0.8f, -0.8f,  0.3f),
                        vec3( 0.8f, -0.8f,  0.3f),
                        vec3( 0.8f,  0.2f,  0.3f),
                        vec3(-0.8f,  0.2f,  0.3f)
                        );
            }

            int index = i - selected;

#define sign(x) (((0) < (x)) - ((x) < (0)))
            auto base_trans = sign(index) * 1.5f;
            auto base_rot   = sign(index) * constants::pi<float>() / 2.5f;
#undef sign

            auto t = mat4::identity();
            t = translate(t, vec3(base_trans + index * 0.2f, 0.0f, 0.0f));

            auto axis = cartesian_axis<3>(cartesian_axis<3>::Y);
            auto r = mat4::identity();
            r = rotate(r, to_vector(axis), -base_rot);

            primitives.back().set_transform(t * r);
        }


        // set prim_id to identify the primitives
        unsigned prim_id = 0;

        // set geom_id to map geometry to materials and textures
        unsigned geom_id = 0;


        // floor
        // Add a default material
        plastic<float> mat;
        mat.set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
        mat.set_cd( from_rgb(0.5f, 0.5f, 0.5f) );
        mat.set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
        mat.set_ka( 1.0f );
        mat.set_kd( 1.0f );
        mat.set_ks( 1.0f );
        mat.set_specular_exp( 32.0f );
        materials.push_back(mat);


        textures.emplace_back(0, 0);

        image tga;
        tga.load("/Users/stefan/visionaray/src/examples/texture/rechenkaestchen.tga");
        auto data_ptr = reinterpret_cast<vector<4, unorm<8>> const*>(tga.data());

        for (size_t i = 0; i < primitives.size(); ++i)
        {
            auto& prim = primitives.at(i);

            prim.prim_id = prim_id++;
            prim.geom_id = geom_id++;


            if (i >= 1)
            {
                // Add a default material
                plastic<float> mat;
                mat.set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
                mat.set_cd( from_rgb(1.0f, 1.0f, 1.0f) );
                mat.set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
                mat.set_ka( 1.0f );
                mat.set_kd( 1.0f );
                mat.set_ks( 1.0f );
                mat.set_specular_exp( 32.0f );
                materials.push_back(mat);
            }


            // textures

            tex_filter_mode fm[] = { Nearest, Linear, CardinalSpline, BSpline };

            if (i >= 1 && i < 5)
            {
                int index = i - 1;
                textures.emplace_back(8, 1);
                auto data = texture_ex::make_rainbow();
                textures.back().set_data(reinterpret_cast<vector<3, unorm<8>> const*>(data.data()));
                textures.back().set_filter_mode(fm[index]);
            }
            else if (i >= 5 && i < 9)
            {
                int index = i - 5;
                textures.emplace_back(16, 16);
                auto data = texture_ex::make_checkerboard();
                textures.back().set_data(reinterpret_cast<vector<3, unorm<8>> const*>(data.data()));
                textures.back().set_address_mode(Wrap);
                textures.back().set_filter_mode(fm[index]);
            }
            else if (i >= 9 && i < 13)
            {
                int index = i - 9;
                textures.emplace_back(1024, 1024);
                textures.back().set_data(data_ptr, PF_RGBA8, PF_RGB8, PremultiplyAlpha);
                textures.back().set_address_mode(Mirror);
                textures.back().set_filter_mode(fm[index]);

            }
            else if (i >= 13 && i <= num_quads)
            {
                int index = i - 13;
                textures.emplace_back(1, 1);
                auto data = texture_ex::make_mandel();
                textures.back().set_data(reinterpret_cast<vector<3, unorm<8>> const*>(data.data()));
                textures.back().set_address_mode(Clamp);
                textures.back().set_filter_mode(fm[index]);

            }
        }


        // assign normals

        // floor
        normals.emplace_back(0.0f, 1.0f, 0.0f);

        // textured quads
        for (size_t i = 1; i < primitives.size(); ++i)
        {
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }


        // assign tex coords

        // floor
        tex_coords.emplace_back(0.0f, 0.0f);
        tex_coords.emplace_back(0.0f, 0.0f);
        tex_coords.emplace_back(0.0f, 0.0f);
        tex_coords.emplace_back(0.0f, 0.0f);

        // textured quads
        for (size_t i = 1; i < primitives.size(); ++i)
        {
            if (i >= 9 && i < 13)
            {
                tex_coords.emplace_back(-1.0f, -1.0f);
                tex_coords.emplace_back( 2.0f, -1.0f);
                tex_coords.emplace_back( 2.0f,  2.0f);
                tex_coords.emplace_back(-1.0f,  2.0f);
            }
            else
            {
                tex_coords.emplace_back(0.0f, 0.0f);
                tex_coords.emplace_back(1.0f, 0.0f);
                tex_coords.emplace_back(1.0f, 1.0f);
                tex_coords.emplace_back(0.0f, 1.0f);
            }
        }
    }

    aabb                                        bbox;
    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // rendering data

    std::vector<texture_ex::quad>               primitives;
    std::vector<vec3>                           normals;
    std::vector<vec2>                           tex_coords;
    std::vector<plastic<float>>                 materials;
    std::vector<texture_type>                   textures;

    int selected    = 5;
    int num_quads   = 15;

protected:

    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

    auto sparams = make_sched_params(
            cam,
            host_rt
            );


    // a light positioned slightly distant above the scene
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position( vec3(0.0f, 10.0f, 10.0f) );

    std::vector<point_light<float>> lights;
    lights.push_back(light);

    auto kparams = make_kernel_params(
            normals_per_face_binding{},
            primitives.data(),
            primitives.data() + primitives.size(),
            normals.data(),
            tex_coords.data(),
            materials.data(),
            textures.data(),
            lights.data(),
            lights.data() + lights.size(),
            2,                          // number of reflective bounces
            0.0001                      // epsilon to avoid self intersection by secondary rays
            );

    auto kernel = whitted::kernel<decltype(kparams)>();
    kernel.params = kparams;

    host_sched.frame(kernel, sparams);

    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}


void renderer::on_key_press(visionaray::key_event const& event)
{
    if (event.key() == keyboard::ArrowLeft || event.key() == keyboard::ArrowDown)
    {
        selected = max(0, selected - 1);
        make_frame();
    }
    else if (event.key() == keyboard::ArrowRight || event.key() == keyboard::ArrowUp)
    {
        selected = min(num_quads - 1, selected + 1);
        make_frame();
    }
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
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

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
