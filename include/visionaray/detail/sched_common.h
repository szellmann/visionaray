// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SCHED_COMMON_H
#define VSNRAY_DETAIL_SCHED_COMMON_H

#include <chrono>

#include <visionaray/mc.h>
#include <visionaray/pixel_format.h>
#include <visionaray/result_record.h>
#include <visionaray/scheduler.h>

#include "color_conversion.h"
#include "macros.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// pixel iteration
//

template <typename T>
struct packet_size
{
    enum { w = 1, h = 1 };
};

template <>
struct packet_size<simd::float4>
{
    enum { w = 2, h = 2 };
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
template <>
struct packet_size<simd::float8>
{
    enum { w = 4, h = 2 };
};
#endif

template <typename T>
struct pixel
{
    VSNRAY_FUNC inline T x(int x) { return T(x); }
    VSNRAY_FUNC inline T y(int y) { return T(y); }
};

template <>
struct pixel<simd::float4>
{
    VSNRAY_CPU_FUNC inline simd::float4 x(int x) { return simd::float4(x, x + 1, x, x + 1); }
    VSNRAY_CPU_FUNC inline simd::float4 y(int y) { return simd::float4(y, y, y + 1, y + 1); }
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
template <>
struct pixel<simd::float8>
{
    VSNRAY_CPU_FUNC inline simd::float8 x(int x)
    {
        return simd::float8(x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3);
    }

    VSNRAY_CPU_FUNC inline simd::float8 y(int y)
    {
        return simd::float8(y, y, y, y, y + 1, y + 1, y + 1, y + 1);
    }
};
#endif


//-------------------------------------------------------------------------------------------------
// color access
//

struct color_access
{
    template <typename InputColor, typename OutputColor>
    VSNRAY_FUNC
    static void store(int x, int y, recti const& viewport, InputColor const& color, OutputColor* buffer)
    {
        convert(buffer[y * viewport.w + x], color);
    }

    template <typename OutputColor>
    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, vector<4, simd::float4> const& color, OutputColor* buffer)
    {
        VSNRAY_ALIGN(16) float r[4];
        VSNRAY_ALIGN(16) float g[4];
        VSNRAY_ALIGN(16) float b[4];
        VSNRAY_ALIGN(16) float a[4];

        using simd::store;

        store(r, color.x);
        store(g, color.y);
        store(b, color.z);
        store(a, color.w);

        auto w = packet_size<simd::float4>::w;
        auto h = packet_size<simd::float4>::h;

        for (auto row = 0; row < h; ++row)
        {
            for (auto col = 0; col < w; ++col)
            {
                if (x + col < viewport.w && y + row < viewport.h)
                {
                    auto idx = row * w + col;
                    convert( buffer[(y + row) * viewport.w + (x + col)], vec4(r[idx], g[idx], b[idx], a[idx]) );
                }
            }
        }
    }

    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, vector<4, simd::float4> const& color, vector<4, float>* buffer)
    {
        static_assert( packet_size<simd::float4>::w == 2, "incompatible packet width" );
        static_assert( packet_size<simd::float4>::h == 2, "incompatible packet height" );

        using simd::store;

        auto c = transpose(color);

        if ( x      < viewport.w &&  y      < viewport.h) store( buffer[ y      * viewport.w +  x     ].data(), c.x);
        if ((x + 1) < viewport.w &&  y      < viewport.h) store( buffer[ y      * viewport.w + (x + 1)].data(), c.y);
        if ( x      < viewport.w && (y + 1) < viewport.h) store( buffer[(y + 1) * viewport.w +  x     ].data(), c.z);
        if ((x + 1) < viewport.w && (y + 1) < viewport.h) store( buffer[(y + 1) * viewport.w + (x + 1)].data(), c.w);
    }

    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, simd::float4 const& value, float* buffer)
    {
        VSNRAY_ALIGN(32) float v[4];

        simd::store(v, value);

        auto w = packet_size<simd::float4>::w;
        auto h = packet_size<simd::float4>::h;

        for (auto row = 0; row < h; ++row)
        {
            for (auto col = 0; col < w; ++col)
            {
                if (x + col < viewport.w && y + row < viewport.h)
                {
                    convert( buffer[(y + row) * viewport.w + (x + col)], v[row * w + col] );
                }
            }
        }
    }

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    template <typename OutputColor>
    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, vector<4, simd::float8> const& color, OutputColor* buffer)
    {
        VSNRAY_ALIGN(32) float r[8];
        VSNRAY_ALIGN(32) float g[8];
        VSNRAY_ALIGN(32) float b[8];
        VSNRAY_ALIGN(32) float a[8];

        using simd::store;

        store(r, color.x);
        store(g, color.y);
        store(b, color.z);
        store(a, color.w);

        const int w = packet_size<simd::float8>::w;
        const int h = packet_size<simd::float8>::h;

        for (auto row = 0; row < h; ++row)
        {
            for (auto col = 0; col < w; ++col)
            {
                if (x + col < viewport.w && y + row < viewport.h)
                {
                    auto idx = row * w + col;
                    convert( buffer[(y + row) * viewport.w + (x + col)], vec4(r[idx], g[idx], b[idx], a[idx]) );
                }
            }
        }
    }
#endif

    template <typename T, typename OutputColor>
    VSNRAY_FUNC
    static void store(int x, int y, recti const& viewport, result_record<T> const& rr, OutputColor* buffer)
    {
        store(x, y, viewport, rr.color, buffer);
    }

    template <typename T, typename Color, typename Depth>
    VSNRAY_FUNC
    static void store(
            int                     x,
            int                     y,
            recti const&            viewport,
            result_record<T> const& rr,
            Color*                  color_buffer,
            Depth*                  depth_buffer
            )
    {
        store(x, y, viewport, rr.color, color_buffer);
        store(x, y, viewport, rr.depth, depth_buffer);
    }


    template <typename InputColor, typename OutputColor>
    VSNRAY_FUNC
    static void get(int x, int y, recti const& viewport, InputColor& color, OutputColor const* buffer)
    {
        convert(color, buffer[y * viewport.w + x]);
    }

    template <typename OutputColor>
    VSNRAY_CPU_FUNC
    static void get(int x, int y, recti const& viewport, vector<4, simd::float4>& color, OutputColor const* buffer)
    {
        static_assert( packet_size<simd::float4>::w == 2, "incompatible packet width" );
        static_assert( packet_size<simd::float4>::h == 2, "incompatible packet height" );

        auto c00 = ( x      < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w +  x     ] : OutputColor();
        auto c01 = ((x + 1) < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w + (x + 1)] : OutputColor();
        auto c10 = ( x      < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w +  x     ] : OutputColor();
        auto c11 = ((x + 1) < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w + (x + 1)] : OutputColor();

        color = simd::pack(vec4(c00), vec4(c01), vec4(c10), vec4(c11));
    }

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    template <typename OutputColor>
    VSNRAY_CPU_FUNC
    static void get(int x, int y, recti const& viewport, vector<4, simd::float8>& color, OutputColor const* buffer)
    {
        const int w = packet_size<simd::float8>::w;
        const int h = packet_size<simd::float8>::h;

        static_assert(w * h == 8, "invalid packet size");

        vec4 v[w * h];

        for (auto row = 0; row < h; ++row)
        {
            for (auto col = 0; col < w; ++col)
            {
                if (x + col < viewport.w && y + row < viewport.h)
                {
                    auto idx = row * w + col;
                    v[idx] = buffer[(y + row) * viewport.w + (x + col)];
                }
            }
        }

        color = vector<4, simd::float8>(
            simd::float8(v[0].x, v[1].x, v[2].x, v[3].x, v[4].x, v[5].x, v[6].x, v[7].x),
            simd::float8(v[0].y, v[1].y, v[2].y, v[3].y, v[4].y, v[5].y, v[6].y, v[7].y),
            simd::float8(v[0].z, v[1].z, v[2].z, v[3].z, v[4].z, v[5].z, v[6].z, v[7].z),
            simd::float8(v[0].w, v[1].w, v[2].w, v[3].w, v[4].w, v[5].w, v[6].w, v[7].w)
            );
    }
#endif

    template <typename InputColor, typename OutputColor, typename T>
    VSNRAY_FUNC
    static void blend(int x, int y, recti const& viewport, InputColor const& color, OutputColor* buffer, T sfactor, T dfactor)
    {
        InputColor dst;

        get(x, y, viewport, dst, buffer);

        dst = color * sfactor + dst * dfactor;

        store(x, y, viewport, dst, buffer);
    }

    template <typename S, typename OutputColor, typename T>
    VSNRAY_FUNC
    static void blend(
            int                     x,
            int                     y,
            recti const&            viewport,
            result_record<S> const& rr,
            OutputColor*            color_buffer,
            T sfactor,
            T dfactor
            )
    {
        blend(x, y, viewport, rr.color, color_buffer, sfactor, dfactor);
    }

    template <typename S, typename OutputColor, typename Depth, typename T>
    VSNRAY_FUNC
    static void blend(
            int                     x,
            int                     y,
            recti const&            viewport,
            result_record<S> const& rr,
            OutputColor*            color_buffer,
            Depth*                  depth_buffer,
            T                       sfactor,
            T                       dfactor
            )
    {
        blend(x, y, viewport, rr.color, color_buffer, sfactor, dfactor);
        blend(x, y, viewport, rr.depth, depth_buffer, sfactor, dfactor);
    }
};


//-------------------------------------------------------------------------------------------------
// calc seed from pixel(s) and return initialized sampler
//

// https://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu

VSNRAY_FUNC
inline unsigned hash(unsigned a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

VSNRAY_FUNC
inline unsigned tic()
{
#if defined(__CUDA_ARCH__)
    return clock64();
#else
    auto t = std::chrono::high_resolution_clock::now();
    return t.time_since_epoch().count();
#endif
}

template <template <typename> class S, typename T>
class sampler_gen
{
public:

    sampler_gen() = default;
    sampler_gen(unsigned seed) : seed_(seed) {}

    VSNRAY_CPU_FUNC S<T> operator()() const
    {
        return S<T>(seed_);
    }

private:

    unsigned seed_ = 0;

};

template <template <typename> class S>
class sampler_gen<S, float>
{
public:

    VSNRAY_FUNC sampler_gen() {}
    VSNRAY_FUNC sampler_gen(unsigned seed) : seed_(seed) {}

    VSNRAY_FUNC S<float> operator()() const
    {
#if defined(__CUDA_ARCH__)
        unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned w = blockDim.x * gridDim.x;
        return S<float>( hash(seed_ + y * w + x) );
#else
        return S<float>(seed_);
#endif
    }

private:

    unsigned seed_ = 0;

};


//-------------------------------------------------------------------------------------------------
// primary ray generators
//

template <typename R, typename T, typename V>
VSNRAY_FUNC
inline R ray_gen(
        T                   x,
        T                   y,
        V const&            viewport,
        vector<3, T> const& eye,
        vector<3, T> const& cam_u,
        vector<3, T> const& cam_v,
        vector<3, T> const& cam_w
        )
{
    auto u = T(2.0) * (x + T(0.5)) / T(viewport.w) - T(1.0);
    auto v = T(2.0) * (y + T(0.5)) / T(viewport.h) - T(1.0);

    R r;
    r.ori = eye;
    r.dir = normalize(cam_u * u + cam_v * v + cam_w);
    return r;
}

template <typename R, typename T, typename V>
VSNRAY_FUNC
inline R ray_gen(
        T                       x,
        T                       y,
        V const&                viewport,
        matrix<4, 4, T> const&  view_matrix,
        matrix<4, 4, T> const&  inv_view_matrix,
        matrix<4, 4, T> const&  proj_matrix,
        matrix<4, 4, T> const&  inv_proj_matrix
        )
{
    VSNRAY_UNUSED(view_matrix);
    VSNRAY_UNUSED(proj_matrix);

    auto u = T(2.0) * (x + T(0.5)) / T(viewport.w) - T(1.0);
    auto v = T(2.0) * (y + T(0.5)) / T(viewport.h) - T(1.0);

    auto o = inv_view_matrix * ( inv_proj_matrix * vector<4, T>(u, v, -1,  1) );
    auto d = inv_view_matrix * ( inv_proj_matrix * vector<4, T>(u, v,  1,  1) );

    R r;
    r.ori =            o.xyz() / o.w;
    r.dir = normalize( d.xyz() / d.w - r.ori );
    return r;
}

template <typename R, typename ...Args>
VSNRAY_FUNC
inline R uniform_ray_gen(unsigned x, unsigned y, Args const&... args)
{
    typedef typename R::scalar_type scalar_type;
    return ray_gen<R>(pixel<scalar_type>().x(x), pixel<scalar_type>().y(y), args...);
}

template <typename R, typename S, typename ...Args>
VSNRAY_FUNC
inline R jittered_ray_gen(unsigned x, unsigned y, S& sampler, Args const&... args)
{
    typedef typename R::scalar_type scalar_type;

    vector<2, scalar_type> jitter
    (
        sampler.next() - scalar_type(0.5),
        sampler.next() - scalar_type(0.5)
    );

    return ray_gen<R>
    (
        pixel<scalar_type>().x(x) + jitter.x,
        pixel<scalar_type>().y(y) + jitter.y,
        args...
    );
}


//-------------------------------------------------------------------------------------------------
// Depth transform
//

template <typename T>
VSNRAY_FUNC inline T depth_transform(
        vector<3, T> const&     isect_pos,
        matrix<4, 4, T> const&  view_matrix,
        matrix<4, 4, T> const&  inv_view_matrix,
        matrix<4, 4, T> const&  proj_matrix,
        matrix<4, 4, T> const&  inv_proj_matrix
        )
{
    VSNRAY_UNUSED(inv_view_matrix);
    VSNRAY_UNUSED(inv_proj_matrix);

    auto pos4 = proj_matrix * (view_matrix * vector<4, T>(isect_pos, T(1.0)));
    auto pos3 = pos4.xyz() / pos4.w;

    return (pos3.z + T(1.0)) * T(0.5);
}


//-------------------------------------------------------------------------------------------------
// Simple uniform pixel sampler
//

template <typename R, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::uniform_type, Args const&... args)
{
    VSNRAY_UNUSED(frame);

    auto r = uniform_ray_gen<R>(x, y, viewport, args...);
    auto result = kernel(r);
    color_access::store(x, y, viewport, result, color_buffer);
}

template <typename R, typename V, typename C, typename D, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(
        unsigned                        x,
        unsigned                        y,
        unsigned                        frame,
        V const&                        viewport,
        C*                              color_buffer,
        D*                              depth_buffer,
        K                               kernel,
        pixel_sampler::uniform_type,
        Args const&...                  args
        )
{
    VSNRAY_UNUSED(frame);

    auto r = uniform_ray_gen<R>(x, y, viewport, args...);
    auto result = kernel(r);
    result.depth = select( result.hit, depth_transform(result.isect_pos, args...), typename R::scalar_type(1.0) );
    color_access::store(x, y, viewport, result, color_buffer, depth_buffer);
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, sampler is passed to kernel
//

template <typename R, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::jittered_type, Args const&... args)
{
    VSNRAY_UNUSED(frame);

    using S = typename R::scalar_type;

    auto gen                            = sampler_gen<sampler, S>();
#if defined(__CUDACC__)
    auto s                              = gen();
#else
    static VSNRAY_THREAD_LOCAL auto s   = gen();
#endif
    auto r                              = jittered_ray_gen<R>(x, y, s, viewport, args...);
    auto result                         = kernel(r, s);
    color_access::store(x, y, viewport, result, color_buffer);
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, result is blended on top of color buffer, sampler is passed to kernel
//

template <typename R, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::jittered_blend_type, Args const&... args)
{
    using S     = typename R::scalar_type;
    using Vec4  = vector<4, S>;

    auto gen                            = sampler_gen<sampler, S>(tic());
#if defined(__CUDACC__)
    auto s                              = gen();
#else
    static VSNRAY_THREAD_LOCAL auto s   = gen();
#endif
    auto r                              = jittered_ray_gen<R>(x, y, s, viewport, args...);
    auto result                         = kernel(r, s);
    auto alpha                          = S(1.0) / S(frame);
    if (frame <= 1)
    {//TODO: clear method in render target?
        color_access::store(x, y, viewport, Vec4(0.0, 0.0, 0.0, 0.0), color_buffer);
    }
    color_access::blend(x, y, viewport, result, color_buffer, alpha, S(1.0) - alpha);
}

template <typename R, typename V, typename C, typename D, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(
        unsigned                            x,
        unsigned                            y,
        unsigned                            frame,
        V const&                            viewport,
        C*                                  color_buffer,
        D*                                  depth_buffer,
        K                                   kernel,
        pixel_sampler::jittered_blend_type  /* */,
        Args const&...                      args
        )
{
    using S     = typename R::scalar_type;

    auto gen                            = sampler_gen<sampler, S>(tic());
#if defined(__CUDACC__)
    auto s                              = gen();
#else
    static VSNRAY_THREAD_LOCAL auto s   = gen();
#endif
    auto r                              = jittered_ray_gen<R>(x, y, s, viewport, args...);
    auto result                         = kernel(r, s);
    auto alpha                          = S(1.0) / S(frame);

    result.depth = select( result.hit, depth_transform(result.isect_pos, args...), typename R::scalar_type(1.0) );

    if (frame <= 1)
    {//TODO: clear method in render target?
        color_access::store(x, y, viewport, result, color_buffer, depth_buffer);
    }
    color_access::blend(x, y, viewport, result, color_buffer, depth_buffer, alpha, S(1.0) - alpha);
}


//-------------------------------------------------------------------------------------------------
// jittered pixel sampler, blends several samples at once
//

template <typename R, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame_begin, unsigned frame_end, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::jittered_blend_type, Args const&... args)
{
    using S     = typename R::scalar_type;
    using Vec4  = vector<4, S>;

    auto gen                            = sampler_gen<sampler, S>(tic());
#if defined(__CUDACC__)
    auto s                              = gen();
#else
    static VSNRAY_THREAD_LOCAL auto s   = gen();
#endif

    for (size_t frame = frame_begin; frame < frame_end; ++frame)
    {
        if (frame <= 1)
        {//TODO: clear method in render target?
            color_access::store(x, y, viewport, Vec4(0.0, 0.0, 0.0, 0.0), color_buffer);
        }
        auto r     = jittered_ray_gen<R>(x, y, s, viewport, args...);
        auto src   = kernel(r, s);
        Vec4 dst;
        color_access::get(x, y, viewport, dst, color_buffer);
        auto alpha = S(1.0) / S(frame);
        dst = dst * S(1 - alpha) + src * S(alpha);
        color_access::store(x, y, viewport, dst, color_buffer);
    }
}

} // detail

//-------------------------------------------------------------------------------------------------
// Dispatch pixel sampler
//

template <typename R, typename V, typename RenderTarget, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(
        R*,
        unsigned        x,
        unsigned        y,
        unsigned        frame,
        V const&        viewport,
        RenderTarget&   rt,
        Args const&...  args
        )
{
    detail::sample_pixel<R>(x, y, frame, viewport, rt.color(), rt.depth(), args...);
}

template <typename R, typename V, template <pixel_format, pixel_format> class RT, pixel_format ColorFormat, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(
        R*,
        unsigned                            x,
        unsigned                            y,
        unsigned                            frame,
        V const&                            viewport,
        RT<ColorFormat, PF_UNSPECIFIED>&    rt,
        Args const&...                      args
        )
{
    detail::sample_pixel<R>(x, y, frame, viewport, rt.color(), args...);
}


} // visionaray

#endif // VSNRAY_DETAIL_SCHED_COMMON_H
