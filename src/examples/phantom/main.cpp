// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include <boost/filesystem.hpp>

#include <GL/glew.h>

#include <pbrtParser/Scene.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/math/io.h>

#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/scheduler.h>
#include <visionaray/thin_lens_camera.h>

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
// Phantom Ray-Hair Intersector (Reshetov and Luebke, 2018)
//

//-------------------------------------------------------------------------------------------------
// Ray/cone intersection from appendix A
//

struct RayConeIntersection
{
    inline bool intersect(float r, float dr)
    {
        float r2  = r * r;
        float drr = r * dr;

        float ddd = cd.x * cd.x + cd.y * cd.y;
        dp        = c0.x * c0.x + c0.y * c0.y;
        float cdd = c0.x * cd.x + c0.y * cd.y;
        float cxd = c0.x * cd.y - c0.y * cd.x;

        float c = ddd;
        float b = cd.z * (drr - cdd);
        float cdz2 = cd.z * cd.z;
        ddd += cdz2;
        float a = 2.0f * drr * cdd + cxd * cxd - ddd * r2 + dp * cdz2;

        float discr = b * b - a * c;
        s   = (b - (discr > 0.0f ? sqrtf(discr) : 0.0f)) / c;
        dt  = (s * cd.z - cdd) / ddd;
        dc  = s * s + dp;
        sp  = cdd / cd.z;
        dp += sp * sp;

        return discr > 0.0f;
    }

    vec3  c0;
    vec3  cd;
    float s;
    float dt;
    float dp;
    float dc;
    float sp;
};


inline bool intersectCylinder(basic_ray<float> const& ray, vec3 p0, vec3 p1, float ra)
{
    vec3  ba = p1 - p0;
    vec3  oc = ray.ori - p0;

    float baba = dot(ba, ba);
    float bard = dot(ba, ray.dir);
    float baoc = dot(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot(oc, ray.dir) - baoc * bard;
    float k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;

    float h = k1 * k1 - k2 * k0;

    if (h < 0.0f)
    {
        return false;
    }

    h = sqrtf(h);
    float t = (-k1 - h) / k2;

    // body
    float y = baoc + t * bard;
    if (y > 0.0f && y < baba)
    {
        return true;
    }

    // caps
    t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;
    if (fabsf(k1 + k2 * t) < h)
    {
        return true;
    }

    return false;
}


//-------------------------------------------------------------------------------------------------
// Curve class
//

struct Curve : primitive<unsigned>
{
    vec3 w0;
    vec3 w1;
    vec3 w2;
    vec3 w3;

    float r;

    vec3 f(float t) const
    {
        float tinv = 1.0f - t;
        return tinv * tinv * tinv * w0
         + 3.0f * tinv * tinv * t * w1
            + 3.0f * tinv * t * t * w2
                      + t * t * t * w3;
    }

    vec3 dfdt(float t) const
    {
        float tinv = 1.0f - t;
        return                 -3.0f * tinv * tinv * w0
         + 3.0f * (3.0f * t * t - 4.0f * t + 1.0f) * w1
                    + 3.0f * (2.0f - 3.0f * t) * t * w2
                                    + 3.0f * t * t * w3;
    }
};

Curve make_curve(vec3 const& w0, vec3 const& w1, vec3 const& w2, vec3 const& w3, float r)
{
    Curve curve;
    curve.w0 = w0;
    curve.w1 = w1;
    curve.w2 = w2;
    curve.w3 = w3;
    curve.r = r;
    return curve;
}


// That's from here: https://www.shadertoy.com/view/MdKBWt
aabb get_bounds(Curve const& curve)
{
    vec3 p0 = curve.w0;
    vec3 p1 = curve.w1;
    vec3 p2 = curve.w2;
    vec3 p3 = curve.w3;

    // extremes
    vec3 mi = min(p0,p3);
    vec3 ma = max(p0,p3);

    // note pascal triangle coefficnets
    vec3 c = -1.0f*p0 + 1.0f*p1;
    vec3 b =  1.0f*p0 - 2.0f*p1 + 1.0f*p2;
    vec3 a = -1.0f*p0 + 3.0f*p1 - 3.0f*p2 + 1.0f*p3;

    vec3 h = b*b - a*c;

    // real solutions
    if (h.x > 0.0f || h.y > 0.0f || h.z > 0.0f)
    {
        vec3 g(
            sqrtf(fabsf(h.x)),
            sqrtf(fabsf(h.y)),
            sqrtf(fabsf(h.z))
            );
        vec3 t1 = clamp((-b - g)/a,vec3(0.0f),vec3(1.0f)); vec3 s1 = 1.0f-t1;
        vec3 t2 = clamp((-b + g)/a,vec3(0.0f),vec3(1.0f)); vec3 s2 = 1.0f-t2;
        vec3 q1 = s1*s1*s1*p0 + 3.0f*s1*s1*t1*p1 + 3.0f*s1*t1*t1*p2 + t1*t1*t1*p3;
        vec3 q2 = s2*s2*s2*p0 + 3.0f*s2*s2*t2*p1 + 3.0f*s2*t2*t2*p2 + t2*t2*t2*p3;

        if (h.x > 0.0f)
        {
            mi.x = min(mi.x,min(q1.x,q2.x));
            ma.x = max(ma.x,max(q1.x,q2.x));
        }
        if (h.y > 0.0f)
        {
            mi.y = min(mi.y,min(q1.y,q2.y));
            ma.y = max(ma.y,max(q1.y,q2.y));
        }
        if (h.z > 0.0f)
        {
            mi.z = min(mi.z,min(q1.z,q2.z));
            ma.z = max(ma.z,max(q1.z,q2.z));
        }
    }

    return aabb(mi - vec3(curve.r), ma + vec3(curve.r));
}


std::pair<Curve, Curve> split(Curve const& curve)
{
    std::pair<Curve, Curve> result;

    vec3 p0 = curve.w0;
    vec3 p1 = curve.w1;
    vec3 p2 = curve.w2;
    vec3 p3 = curve.w3;

    vec3 q0 = (p0 + p1) / 2.0f;
    vec3 q1 = (p1 + p2) / 2.0f;
    vec3 q2 = (p2 + p3) / 2.0f;

    vec3 r0 = (q0 + q1) / 2.0f;
    vec3 r1 = (q1 + q2) / 2.0f;

    vec3 s0 = (r0 + r1) / 2.0f;

    result.first  = make_curve(p0, q0, r0, s0, curve.r);
    result.second = make_curve(s0, r1, q2, p3, curve.r);

    return result;
}


//-------------------------------------------------------------------------------------------------
// Transform entities to ray-centric coordinate system RCC
//

struct TransformToRCC
{
    inline TransformToRCC(basic_ray<float> const& r)
    {
        vec3 e1;
        vec3 e2;
        vec3 e3 = normalize(r.dir);
        make_orthonormal_basis(e1, e2, e3);
        xformInv = mat4(
            vec4(e1,    0.0f),
            vec4(e2,    0.0f),
            vec4(e3,    0.0f),
            vec4(r.ori, 1.0f)
            );
        xform = inverse(xformInv);
    }

    inline vec3 xfmPoint(vec3 point)
    {
        return (xform * vec4(point, 1.0f)).xyz();
    }

    inline vec3 xfmVector(vec3 vector)
    {
        return (xform * vec4(vector, 0.0f)).xyz();
    }

    inline vec3 xfmPointInv(vec3 point)
    {
        return (xformInv * vec4(point, 1.0f)).xyz();
    }

    inline vec3 xfmVectorInv(vec3 vector)
    {
        return (xformInv * vec4(vector, 0.0f)).xyz();
    }

    mat4 xform;
    mat4 xformInv;
};

inline hit_record<basic_ray<float>, primitive<unsigned>> intersect(basic_ray<float> const& r, Curve const& curve)
{
    hit_record<basic_ray<float>, primitive<unsigned>> result;
    result.hit = false;

    // Early exit check against enclosing cylinder
    auto distToCylinder = [&curve](vec3 pt) {
        return length(cross(pt - curve.w0, pt - curve.w3)) / length(curve.w3 - curve.w0);
    };

    // TODO: could compute tighter bounding cylinder than this one!
    float rmax = distToCylinder(curve.f(0.33333f));
    rmax = fmaxf(rmax, distToCylinder(curve.f(0.66667f)));
    rmax += curve.r;

    vec3 axis = normalize(curve.w3 - curve.w0);
    vec3 p0   = curve.w0 - axis * curve.r;
    vec3 p1   = curve.w3 + axis * curve.r;

    if (!intersectCylinder(r, p0, p1, rmax))
    {
        return result;
    }

    // Transform curve to RCC
    TransformToRCC rcc(r);
    Curve xcurve = make_curve(
        rcc.xfmPoint(curve.w0),
        rcc.xfmPoint(curve.w1),
        rcc.xfmPoint(curve.w2),
        rcc.xfmPoint(curve.w3),
        curve.r
        );

    // "Test for convergence. If the intersection is found,
    // report it, otherwise start at the other endpoint."

    // Compute curve end to start at
    float tstart = dot(xcurve.w3 - xcurve.w0, r.dir) > 0.0f ? 0.0f : 1.0f;

    for (int ep = 0; ep < 2; ++ep)
    {
        float t   = tstart;

        RayConeIntersection rci;

        float told = 0.0f;
        float dt1 = 0.0f;
        float dt2 = 0.0f;

        for (int i = 0; i < 40; ++i)
        {
            rci.c0 = xcurve.f(t);
            rci.cd = xcurve.dfdt(t);

            bool phantom = !rci.intersect(curve.r, 0.0f/*cylinder*/);

            // "In all examples in this paper we stop iterations when dt < 5x10^−5"
            if (!phantom && fabsf(rci.dt) < 5e-5f)
            {
                //vec3 n = normalize(curve.dfdt(t));
                rci.s += rci.c0.z;
                result.t = rci.s;
                result.u = t; // abuse param u to store curve's t
                result.hit = true;
                result.isect_pos = r.ori + result.t * r.dir;
                break;
            }

            rci.dt = min(rci.dt, 0.5f);
            rci.dt = max(rci.dt, -0.5f);

            dt1 = dt2;
            dt2 = rci.dt;

            // Regula falsi
            if (dt1 * dt2 < 0.0f)
            {
                float tnext = 0.0f;
                // "we use the simplest possible approach by switching
                // to the bisection every 4th iteration:"
                if ((i & 3) == 0)
                {
                    tnext = 0.5f * (told + t);
                }
                else
                {
                    tnext = (dt2 * told - dt1 * t) / (dt2 - dt1);
                }
                told = t;
                t = tnext;
            }
            else
            {
                told = t;
                t += rci.dt;
            }

            if (t < 0.0f || t > 1.0f)
            {
                break;
            }
        }

        if (!result.hit)
        {
            tstart = 1.0f - tstart;
        }
        else
        {
            break;
        }
    }

    return result;
}

inline vec3 get_normal(
        hit_record<basic_ray<float>, primitive<unsigned>> const& hr,
        Curve const& curve
        )
{
    float t = hr.u;
    vec3 curve_pos = curve.f(t);
    return normalize(hr.isect_pos - curve_pos);
}

inline float area(Curve const& curve)
{
    VSNRAY_UNUSED(curve);

    // TODO: implement this to support curve lights!
    return -1.0f;
}

inline void split_primitive(aabb& L, aabb& R, float plane, int axis, Curve const& curve)
{
    VSNRAY_UNUSED(L);
    VSNRAY_UNUSED(R);
    VSNRAY_UNUSED(plane);
    VSNRAY_UNUSED(axis);
    VSNRAY_UNUSED(curve);

    // TODO: implement this to support SBVHs
}


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
// struct with state variables
//

struct renderer : viewer_type
{
    renderer()
        : viewer_type(512, 512, "Visionaray Phantom Ray-Hair Intersector Example")
        , host_sched(8)
    {
        using namespace support;

        // Add cmdline options
        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "filename",
            cl::Desc("Input file in wavefront obj format"),
            cl::Positional,
            cl::Optional,
            cl::init(this->filename)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "camera",
            cl::Desc("Text file with camera parameters"),
            cl::ArgRequired,
            cl::init(this->initial_camera)
            ) );
    }

    void build_scene()
    {
        std::vector<Curve> actual_curves;

        std::shared_ptr<pbrt::Scene> scene;

        if (!filename.empty())
        {
            try
            {
                boost::filesystem::path p(filename);
                std::string ext = p.extension().string();

                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                std::cout << "Try loading pbrt file...\n";

                if (ext == ".pbf")
                {
                    scene = pbrt::Scene::loadFrom(filename);
                }
                else if (ext == ".pbrt")
                {
                    scene = pbrt::importPBRT(filename);
                }

                for (pbrt::Shape::SP shape : scene->world->shapes)
                {
                    if (pbrt::Curve::SP curve = std::dynamic_pointer_cast<pbrt::Curve>(shape))
                    {
                        if (curve->P.size() != 4)
                        {
                            continue;
                        }

                        vec3 w0(curve->P[0].x, curve->P[0].y, curve->P[0].z);
                        vec3 w1(curve->P[1].x, curve->P[1].y, curve->P[1].z);
                        vec3 w2(curve->P[2].x, curve->P[2].y, curve->P[2].z);
                        vec3 w3(curve->P[3].x, curve->P[3].y, curve->P[3].z);
                        // TODO: phantom _should_ also support two radii!
                        float r = (curve->width0 + curve->width1) * 0.5f;

                        actual_curves.push_back(make_curve(w0, w1, w2, w3, r));
                    }
                }
            }
            catch (std::runtime_error e)
            {
                std::cout << "Failed: " << e.what() << '\n';
                // ignore
            }
        }

        // Add some dummy data when file couldn't be loaded / file name was empty
        if (actual_curves.empty())
        {
            actual_curves.push_back(make_curve({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.5f}, {2.0f, 0.5f, 0.0f}, {3.0f, 0.0f, 0.0f}, 0.3f));
            actual_curves.push_back(make_curve({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {3.0f, 3.0f, 0.0f}, 0.08f));
            actual_curves.push_back(make_curve({0.0f, 0.0f, 0.0f}, {0.0f, 20.0f, 1.0f}, {0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, 3.0f}, 0.04f));
            actual_curves.push_back(make_curve({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 2.0f}, {0.0f, 0.0f, 3.0f}, 0.08f));
            actual_curves.push_back(make_curve({-1.0f, 2.0f, 0.0f}, {0.0f, -2.0f, 0.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 0.0f}, 0.02f));
            actual_curves.push_back(make_curve({-5.0f, 2.0f, 0.0f}, {0.0f, -0.5f, 1.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 8.0f}, 0.07f));
            actual_curves.push_back(make_curve({0.0f, 10.0f, 0.0f}, {0.0f, 11.0f, 11.0f}, {0.0f, -11.0f, 0.0f}, {0.0f, -10.0f, 0.0f}, 0.3f));
            actual_curves.push_back(make_curve({3.0f, 8.0f, 0.0f}, {0.0f, 15.0f, 11.0f}, {0.0f, -11.0f, 0.0f}, {7.0f, -10.0f, 0.0f}, 0.2f));
            actual_curves.push_back(make_curve({-10.0f, 20.0f, 0.0f}, {0.0f, -20.0f, 0.0f}, {1.0f, 4.0f, 0.0f}, {2.0f, -4.0f, 0.0f}, 0.1f));
            actual_curves.push_back(make_curve({0.0f, 20.0f, 0.0f}, {1.0f, 2.0f, 4.5f}, {2.0f, 3.5f, 0.0f}, {3.0f, 4.0f, 0.0f}, 0.3f));
        }

        for (auto const& curve : actual_curves)
        {
            auto p0 = split(curve);
            auto p00 = split(p0.first);
            auto p01 = split(p0.second);
            auto p000 = split(p00.first);
            auto p001 = split(p00.second);
            auto p010 = split(p01.first);
            auto p011 = split(p01.second);

            // 4x
            curves.push_back(p00.first);
            curves.push_back(p00.second);
            curves.push_back(p01.first);
            curves.push_back(p01.second);

            // 8x
            // curves.push_back(p000.first);
            // curves.push_back(p000.second);
            // curves.push_back(p001.first);
            // curves.push_back(p001.second);
            // curves.push_back(p010.first);
            // curves.push_back(p010.second);
            // curves.push_back(p011.first);
            // curves.push_back(p011.second);
        }

        for (unsigned i = 0; i < curves.size(); ++i)
        {
            curves[i].prim_id = i;
            curves[i].geom_id = 0;
        }

        std::cout << "Curves loaded:          " << actual_curves.size() << '\n';
        std::cout << "Curves after splitting: " << curves.size() << '\n';

        matte<float> m;
        m.cd() = from_rgb(0.78f, 0.70f, 0.55f);
        m.kd() = 1.0f;
        materials.emplace_back(m);

        binned_sah_builder builder;
        // TODO: implement get_split() for Curve to support spatial splits / SBVH!
        builder.enable_spatial_splits(false);

        std::cout << "Building BVH...\n";
        bvh = builder.build(index_bvh<Curve>{}, curves.data(), curves.size());
        std::cout << "Done!\n";

        bbox.invalidate();

        for (auto const& curve : curves)
        {
            bboxes.push_back(get_bounds(curve));
            bbox.insert(get_bounds(curve));
        }
    }

    thin_lens_camera                              cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<basic_ray<float>>               host_sched;

    std::vector<Curve>                          curves;
    std::vector<aabb>                           bboxes;
    std::vector<matte<float>>                   materials;
    index_bvh<Curve>                            bvh;

    aabb                                        bbox;

    unsigned                                    frame_num       = 0;
    vec3                                        ambient         = vec3(1.0f, 1.0f, 1.0f);

    std::string                                 filename;
    std::string                                 initial_camera;

protected:

    void load_camera(std::string filename);
    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Load camera from file, reset frame counter and clear frame
//

void renderer::load_camera(std::string filename)
{
    std::ifstream file(filename);
    if (file.good())
    {
        file >> cam;
        frame_num = 0;
        host_rt.clear_color_buffer();
        std::cout << "Load camera from file: " << filename << '\n';
    }
}


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

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

    // Create bvh "refs" that we can pass to the
    // path tracing kernel
    using bvh_ref = index_bvh<Curve>::bvh_ref;
    aligned_vector<bvh_ref> primitives;
    primitives.push_back(bvh.ref());

    // Construct a parameter object that is
    // compatible with the builtin path tracing kernel.
    auto kparams = make_kernel_params(
            primitives.data(),
            primitives.data() + primitives.size(),
            materials.data(),
            4,      // bounces
            1e-5f,  // scene epsilon
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

    glEnable(GL_FRAMEBUFFER_SRGB);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // You could also directly access host_rt::color()
    // or host_rt::depth() (this render target however
    // doesn't store a depth buffer).
    host_rt.display_color_buffer();
}


void renderer::on_key_press(visionaray::key_event const& event)
{
    static const std::string camera_file_base = "visionaray-camera";
    static const std::string camera_file_suffix = ".txt";

    switch (event.key())
    {
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

    rend.build_scene();

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.set_lens_radius(0.002f);
    rend.cam.set_focal_distance(2.0f);

    // Load camera from file or set view-all
    std::ifstream file(rend.initial_camera);
    if (file.good())
    {
        file >> rend.cam;
    }
    else
    {
        rend.cam.view_all(rend.bbox);
    }

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
