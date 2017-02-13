// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include <visionaray/detail/platform.h>

#include <visionaray/math/simd/sse.h>

#include <visionaray/texture/texture.h>

#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/material.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#ifdef __CUDACC__
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/rotate_manipulator.h>
#include <common/manip/translate_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

using namespace visionaray;

using manipulators  = std::vector<std::shared_ptr<visionaray::camera_manipulator>>;
using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Visionaray multi-volume rendering example
//
//  - If compiled with nvcc, only CUDA code is generated
//  - If compiled with host compiler, x86 code is generated
//
// The example shows the workflow when programming a CUDA-compatible algorithm:
//  - User is responsible of copying data to the GPU.
//  - Built-in data structures w/ interfaces similar to the host are used.
//  - Permanent data is copied to the GPU only once and then referred to using
//    reference objects.
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Texture data
//

// post-classification transfer functions
VSNRAY_ALIGN(32) static const vec4 tfdata[3 * 5] = {

        { 0.0f, 0.2f, 0.8f, 0.005f }, // 1st volume
        { 1.0f, 0.8f, 0.0f, 0.01f  },
        { 0.0f, 1.0f, 0.0f, 0.50f  },
        { 1.0f, 0.8f, 0.0f, 0.01f  },
        { 1.0f, 1.0f, 1.0f, 0.005f },


        { 1.0f, 0.0f, 0.0f, 1.0f   }, // 2nd volume
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 0.0f, 0.0f, 0.2f   },
        { 1.0f, 1.0f, 1.0f, 0.002f },



        { 1.0f, 1.0f, 1.0f, 1.0f   }, // 3rd volume
        { 1.0f, 1.0f, 1.0f, 0.50f  },
        { 1.0f, 1.0f, 1.0f, 0.50f  },
        { 1.0f, 1.0f, 1.0f, 0.50f  },
        { 0.0f, 0.0f, 1.0f, 0.002f }

        };


//-------------------------------------------------------------------------------------------------
// Helpers
//

template <typename T, typename Tex>
VSNRAY_FUNC
vector<3, T> gradient(Tex const& tex, vector<3, T> tex_coord)
{
    vector<3, T> s1;
    vector<3, T> s2;

    float DELTA = 0.01f;

    s1.x = tex3D(tex, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f));
    s2.x = tex3D(tex, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f));
    s1.y = tex3D(tex, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f));
    s2.y = tex3D(tex, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f));
    s1.z = tex3D(tex, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA));
    s2.z = tex3D(tex, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA));

    return s2 - s1;
}


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
#ifdef __CUDACC__
    using ray_type = basic_ray<float>;
#else
    using ray_type = basic_ray<simd::float4>;
#endif
    using model_manipulators = std::vector<std::shared_ptr<model_manipulator>>;

    renderer()
        : viewer_type(512, 512, "Visionaray Multi-Volume Rendering Example")
        , host_sched(8)
    {
    }

    void setup()
    {
        for (size_t i = 0; i < 3; ++i)
        {
            if (i % 3 == 0)
            {
                int w = 201;
                int h = 201;
                int d = 201;

                if (marschner_lobb.size() == 0)
                {
                    make_marschner_lobb(w, h, d);
                }

                volumes.emplace_back(w, h, d);
                auto& volume = volumes.back();
                volume.reset(marschner_lobb.data());
                volume.set_filter_mode(Linear);
                volume.set_address_mode(Clamp);

                bboxes.emplace_back(vec3( -1.0f, -1.0f, -1.0f ), vec3( 1.0f, 1.0f, 1.0f ));
            }
            else if (i % 3 == 1)
            {
                int w = 256;
                int h = 256;
                int d = 256;

                if (heart.size() == 0)
                {
                    make_heart(w, h, d);
                }

                volumes.emplace_back(w, h, d);
                auto& volume = volumes.back();
                volume.reset(heart.data());
                volume.set_filter_mode(Linear);
                volume.set_address_mode(Clamp);

                bboxes.emplace_back(vec3( -1.0f, -1.0f / 1.0f, -1.0f ), vec3( 1.0f, 1.0f / 1.0f, 1.0f ));

            }
            else
            {
                int w = 256;
                int h = 256;
                int d = 256;

                if (mandelbulb.size() == 0)
                {
                    make_mandelbulb(w, h, d);
                }

                volumes.emplace_back(w, h, d);
                auto& volume = volumes.back();
                volume.reset(mandelbulb.data());
                volume.set_filter_mode(Linear);
                volume.set_address_mode(Clamp);

                bboxes.emplace_back(vec3( -1.0f, -1.0f / 1.0f, -1.0f ), vec3( 1.0f, 1.0f / 1.0f, 1.0f ));
            }

            transfuncs.emplace_back(5);
            auto& transfunc = transfuncs.back();
            transfunc.reset(&tfdata[(i % 3) * 5]);
            transfunc.set_filter_mode(Nearest);
            transfunc.set_address_mode(Clamp);

            auto axis = cartesian_axis<3>(cartesian_axis<3>::label(i % 3));
            auto r = mat4::identity();
            r = rotate(r, to_vector(axis), constants::pi<float>() / 4.0f);

            auto s = mat4::identity();
            s = scale(s, vec3(1.0f, 1.0f, 1.0f));

            auto t = mat4::identity();
            t = translate(t, vec3(i * 2.5f, 0.0f, 0.0f));

            transforms.push_back(t * r * s);
        }

        for (size_t i = 0; i < volumes.size(); ++i)
        {
            model_manips.emplace_back( std::make_shared<rotate_manipulator>(
                    cam,
                    transforms[i],
                    vec3(2.0f),
                    mouse::Left
                    ) );

            model_manips.emplace_back( std::make_shared<translate_manipulator>(
                    cam,
                    transforms[i],
                    vec3(2.0f),
                    mouse::Left
                    ) );

        }
    }

    void make_marschner_lobb(int w, int h, int d)
    {
        marschner_lobb.resize(w * h * d);

        float fM = 6.0f;
        float a  = 0.25f;

        auto rho = [=](float r) -> float
        {
            return cos(2 * constants::pi<float>() * fM * cos(constants::pi<float>() * r / 2));
        };

        for (int z = 0; z < d; ++z)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    int index = w * h * z + w * y + x;

                    auto xx = (float)x / w * 2.0f - 1.0f;
                    auto yy = (float)y / h * 2.0f - 1.0f;
                    auto zz = (float)z / d * 2.0f - 1.0f;

                    marschner_lobb[index] =
                            (1 - sin(constants::pi<float>() * zz / 2) + a * (1 + rho(length(vec2(xx, yy)))))
                                                    / ( 2 * (1 + a) );
                }
            }
        }
    }

    void make_heart(int w, int h, int d)
    {
        heart.resize(w * h * d);

        for (int z = 0; z < d; ++z)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    int index = w * h * z + w * y + x;

                    auto xx = (float)x / w * 4.0f - 2.0f;
                    auto yy = (float)y / h * 4.0f - 2.0f;
                    auto zz = (float)z / d * 4.0f - 2.0f;

                    heart[index] = pow(xx * xx + (9.0f / 4.0f) * (yy * yy) + zz * zz - 1, 3)
                                 - (xx * xx) * (zz * zz * zz) - (9.0f / 80.0f) * (yy * yy) * (zz * zz * zz);
                }
            }
        }
    }

    void make_mandelbulb(int w, int h, int d)
    {
        mandelbulb.resize(w * h * d);

        float n = 8;

        for (int z = 0; z < d; ++z)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    auto v0 = 3.0f * (vec3(x, y, z) - vec3(w, h, d) / 2.0f) / vec3(w, h, d);
                    auto vv = v0;

                    int i = 0;
                    for (; i < n; ++i)
                    {
                        float r     = length(vv);
                        float theta = atan2(length(vv.xy()), vv.z);
                        float phi   = atan2(vv.y, vv.x);

                        vv = pow(r, static_cast<float>(n)) * vec3(
                                sin(theta * n) * cos(phi * n),
                                sin(theta * n) * sin(phi * n),
                                cos(theta * n)
                                ) + vec3(v0.x, v0.y, 0.0f);

                        if (dot(vv, vv) > 2.0f)
                        {
                            break;
                        }
                    }

                    int index = w * h * z + w * y + x;

                    mandelbulb[index] = i < n;// ? i / static_cast<float>(n) : 0.0f;
                }
            }
        }
    }

    void render_model_manipulators()
    {
        for (auto& manip : model_manips)
        {
            if (manip->active())
            {
                manip->render();
            }
        }
    }

#ifdef __CUDACC__
    void upload_gpu_textures()
    {
        for (auto const& volume : volumes)
        {
            device_volumes_storage.emplace_back(volume);
        }

        for (auto const& transfunc : transfuncs)
        {
            device_transfuncs_storage.emplace_back(transfunc);
        }
    }
#endif

    camera                                                      cam;
    manipulators                                                manips;
    tiled_sched<ray_type>                                       host_sched;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED>                     host_rt;
#ifdef __CUDACC__
    cuda_sched<ray_type>                                        device_sched;
    pixel_unpack_buffer_rt<PF_RGBA8, PF_UNSPECIFIED>            device_rt;
#endif

    model_manipulators                                          model_manips;


    // data

    aligned_vector<float>                                       marschner_lobb;
    aligned_vector<float>                                       heart;
    aligned_vector<float>                                       mandelbulb;


    // textures and texture references

    // On the CPU, we can simply "ref" the arrays with data
    std::vector<texture_ref<float, 3>>                          volumes;
    std::vector<texture_ref<vec4, 1>>                           transfuncs;

#ifdef __CUDACC__
    // On the GPU, we need permanent storage in texture memory
    // and will create references later on
    std::vector<cuda_texture<float, 3>>                         device_volumes_storage;
    std::vector<cuda_texture<vec4, 1>>                          device_transfuncs_storage;
#endif


    // transforms etc.

    std::vector<aabb>                                           bboxes;
    std::vector<mat4>                                           transforms;

protected:

    void on_display();
    void on_key_press(key_event const& event);
    void on_mouse_down(visionaray::mouse_event const& event);
    void on_mouse_up(visionaray::mouse_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// The rendering kernel
// A C++03 functor for compatibility with CUDA versions that don't
// support device lambda functions yet
//

struct kernel
{
    using R    = renderer::ray_type;
    using S    = R::scalar_type;
    using V    = vector<3, S>;
    using C    = vector<4, S>;
    using Mat4 = matrix<4, 4, S>;
    using HR   = hit_record<R, aabb>;


    VSNRAY_GPU_FUNC
    result_record<S> operator()(R ray)
    {
        result_record<S> result;

        // visionaray::numeric_limits is compatible with
        // CUDA and with x86 SIMD types, prefer this in
        // a cross-platform kernel.

        S tmin =  numeric_limits<float>::max();
        S tmax = -numeric_limits<float>::max();


        // tnear and tfar for each volume

        vector<2, S> range[MAX_VOLS];

        for (size_t i = 0; i < num_volumes; ++i)
        {
            R inv_ray;
            inv_ray.ori = (Mat4(transforms_inv[i]) * vector<4, S>(ray.ori, S(1.0))).xyz();
            inv_ray.dir = (Mat4(transforms_inv[i]) * vector<4, S>(ray.dir, S(0.0))).xyz();

            auto hit_rec = intersect(inv_ray, bboxes[i]);

            tmin = select(
                    hit_rec.hit && hit_rec.tnear < tmin,
                    hit_rec.tnear,
                    tmin
                    );

            tmax = select(
                    hit_rec.hit && hit_rec.tfar > tmax,
                    hit_rec.tfar,
                    tmax
                    );

            range[i].x = hit_rec.tnear;
            range[i].y = hit_rec.tfar;
        }

        auto t = tmin;
        auto delta_t = 0.007f;

        result.color = C(0.0);

        while ( visionaray::any(t < tmax) )
        {
            auto color = C(0.0f);

            for (size_t i = 0; i < num_volumes; ++i)
            {
                auto inside = t >= range[i].x && t < range[i].y;

                if (visionaray::any(inside))
                {
                    auto pos = ray.ori + ray.dir * t;
                         pos = (Mat4(transforms_inv[i]) * vector<4, S>(pos, S(1.0f))).xyz();

                    auto tex_coord = vector<3, S>(
                            ( pos.x + 1.0f ) / 2.0f,
                            (-pos.y + 1.0f ) / 2.0f,
                            (-pos.z + 1.0f ) / 2.0f
                            );

                    // sample volume and do post-classification
                    auto voxel = tex3D(volumes[i], tex_coord);
                    C colori = tex1D(transfuncs[i], voxel);


                    auto do_shade = colori.w >= 0.1f;

                    if (visionaray::any(do_shade))
                    {
                        auto grad = gradient(volumes[i], tex_coord);
                        do_shade &= length(grad) != 0.0f;

                        shade_record<decltype(light), S> sr;
                        sr.isect_pos = pos;
                        sr.light = light;
                        sr.normal = normalize(grad);
                        sr.view_dir = -ray.dir;
                        auto light_pos = ( Mat4(transforms_inv[i]) * vector<4, S>(V(sr.light.position()), S(1.0)) ).xyz();
                        sr.light_dir = normalize(light_pos);

                        auto shaded_clr = materials[i].shade(sr);
                        colori.xyz() = mul(
                                colori.xyz(),
                                to_rgb(shaded_clr),
                                do_shade,
                                colori.xyz()
                                );
                    }


                    // opacity correction
//                    colori.w = 1.0f - pow(1.0f - colori.w, delta_t);

                    // premultiplied alpha
                    colori.xyz() *= colori.w;

                    color += select(inside, colori, C(0.0f));
                }
            }


            // front-to-back alpha compositing
            result.color += select(
                    t < tmax,
                    color * (1.0f - result.color.w),
                    C(0.0)
                    );


            // early-ray termination - don't traverse w/o a contribution
            if ( visionaray::all(result.color.w >= 0.999) )
            {
                break;
            }

            // step on
            t += delta_t;
        }

        result.hit = tmax > tmin;
        return result;
    }


    // Kernel parameters: textures, bounding boxes, inverse transforms...

    static const int MAX_VOLS = 32;

    size_t num_volumes;

#ifdef __CUDACC__
    cuda_texture_ref<float, 3> const*   volumes;
    cuda_texture_ref<vec4, 1> const*    transfuncs;
#else
    texture_ref<float, 3> const*        volumes;
    texture_ref<vec4, 1> const*         transfuncs;
#endif

    matrix<4, 4, S> const*              transforms_inv;
    aabb const*                         bboxes;
    plastic<S> const*                   materials;
    point_light<float>                  light;
};


//-------------------------------------------------------------------------------------------------
// Display function, calls the volume rendering kernel
//

void renderer::on_display()
{
    // some setup

    using R = renderer::ray_type;
    using S = R::scalar_type;
    using C = vector<3, S>;

#ifdef __CUDACC__
    auto sparams = make_sched_params(
            cam,
            device_rt
            );
#else
    auto sparams = make_sched_params(
            cam,
            host_rt
            );
#endif


    // setup kernel parameters

#ifdef __CUDACC__
    thrust::device_vector<matrix<4, 4, S>>  param_transforms_inv;
    thrust::device_vector<aabb>             param_bboxes;
    thrust::device_vector<plastic<S>>       param_materials;
#else
    aligned_vector<matrix<4, 4, S>>         param_transforms_inv;
    aligned_vector<aabb>                    param_bboxes;
    aligned_vector<plastic<S>>              param_materials;
#endif

    param_transforms_inv.resize(transforms.size());
    param_bboxes.resize(bboxes.size());
    param_materials.resize(transforms.size());

    for (size_t i = 0; i < transforms.size(); ++i)
    {
        param_transforms_inv[i] = inverse(transforms[i]);
    }

    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        param_bboxes[i] = bboxes[i];
    }

    for (size_t i = 0; i < transforms.size(); ++i)
    {
        plastic<S> mat;
        mat.set_ca( from_rgb(C(0.2f, 0.2f, 0.2f)) );
        mat.set_cd( from_rgb(C(0.8f, 0.8f, 0.8f)) );
        mat.set_cs( from_rgb(C(0.8f, 0.8f, 0.8f)) );
        mat.set_ka( 1.0f );
        mat.set_kd( 1.0f );
        mat.set_ks( 1.0f );
        mat.set_specular_exp( 128.0f );
        param_materials[i] = mat;
    }

    kernel kern;

#ifdef __CUDACC__

    // w/ CUDA, it's the users' responsibility to copy the kernel data to
    // the GPU. thrust provides a convenient, STL-like interface for that.

    thrust::device_vector<cuda_texture_ref<float, 3>> device_volumes;
    thrust::device_vector<cuda_texture_ref<vec4, 1>> device_transfuncs;
    device_volumes.resize(volumes.size());
    device_transfuncs.resize(transfuncs.size());

    using volume_ref = thrust::device_vector<cuda_texture_ref<float, 3>>::value_type;
    using transfunc_ref = thrust::device_vector<cuda_texture_ref<vec4, 1>>::value_type;

    for (size_t i = 0; i < device_volumes_storage.size(); ++i)
    {
        size_t index = i % device_volumes_storage.size();

        device_volumes[i] = volume_ref(device_volumes_storage[index]);
    }

    for (size_t i = 0; i < device_transfuncs.size(); ++i)
    {
        device_transfuncs[i] = transfunc_ref(device_transfuncs_storage[i]);
    }


    kern.num_volumes    = device_volumes.size();
    kern.volumes        = thrust::raw_pointer_cast(device_volumes.data());
    kern.transfuncs     = thrust::raw_pointer_cast(device_transfuncs.data());
    kern.transforms_inv = thrust::raw_pointer_cast(param_transforms_inv.data());
    kern.bboxes         = thrust::raw_pointer_cast(param_bboxes.data());
    kern.materials      = thrust::raw_pointer_cast(param_materials.data());
#else

    // Nothing to copy with x86, just pass along some pointers

    kern.num_volumes    = volumes.size();
    kern.volumes        = volumes.data();
    kern.transfuncs     = transfuncs.data();
    kern.transforms_inv = param_transforms_inv.data();
    kern.bboxes         = param_bboxes.data();
    kern.materials      = param_materials.data();
#endif

    kern.light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    kern.light.set_kl( 1.0f );
    kern.light.set_position( cam.eye() );


    // call kernel in schedulers' frame() method

#ifdef __CUDACC__
    device_sched.frame(kern, sparams);
#else
    host_sched.frame(kern, sparams);
#endif


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef __CUDACC__
    device_rt.display_color_buffer();
#else
    host_rt.display_color_buffer();
#endif

    render_model_manipulators();
}


//-------------------------------------------------------------------------------------------------
// Keyboard handling
//

void renderer::on_key_press(key_event const& event)
{
    if (event.key() == keyboard::r)
    {
        for (auto it = model_manips.begin(); it != model_manips.end(); ++it)
        {
            if ((*it)->active())
            {
                (*it)->set_active(false);
                auto next = ++it;
                if (next != model_manips.end())
                {
                    (*next)->set_active(true);
                }
                return;
            }
        }

        (*model_manips.begin())->set_active(true);
    }

    viewer_base::on_key_press(event);
}


//-------------------------------------------------------------------------------------------------
// Mouse handling
//

void renderer::on_mouse_down(visionaray::mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_down(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_mouse_down(event);
}

void renderer::on_mouse_up(visionaray::mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_up(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_mouse_up(event);
}

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    for (auto& manip : model_manips)
    {
        if (manip->active())
        {
            if (manip->handle_mouse_move(event))
            {
                return;
            }
        }
    }

    // usual handling if no transform manip intercepted
    viewer_base::on_mouse_move(event);
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
#ifdef __CUDACC__
    device_rt.resize(w, h);
#endif

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

    // Initialize volume textures, manipulators, etc.
    rend.setup();

#ifdef __CUDACC__
    rend.upload_gpu_textures();
#endif

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    aabb bbox;
    bbox.invalidate();

    for (size_t i = 0; i < rend.bboxes.size(); ++i)
    {
        auto b = rend.bboxes[i];

        auto verts = compute_vertices(b);

        for (auto v : verts)
        {
            auto v4 = rend.transforms[i] * vec4(v, 1.0f);

            bbox = combine(bbox, v4.xyz());
        }
    }

    rend.cam.view_all( bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
