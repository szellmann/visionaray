// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SCHED_COMMON_H
#define VSNRAY_DETAIL_SCHED_COMMON_H

#include <chrono>

#include <visionaray/mc.h>
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

template <typename CT /* output color traits */, typename CI /* input color */>
struct color_access
{
    typedef CT color_traits;
    typedef typename CT::type output_color;
    typedef CI input_color;

    VSNRAY_FUNC
    static void store(int x, int y, recti const& viewport, input_color const& color, output_color* buffer)
    {
        convert<color_traits::format> cv;
        buffer[y * viewport.w + x] = cv(color);
    }

    template <typename T>
    VSNRAY_FUNC
    static void blend(int x, int y, recti const& viewport, input_color const& color, output_color* buffer, T sfactor, T dfactor)
    {
        auto dst = get(x, y, viewport, buffer);
        dst = color * sfactor + dst * dfactor;
        store(x, y, viewport, dst, buffer);
    }

    VSNRAY_FUNC
    static input_color get(int x, int y, recti const& viewport, output_color* buffer)
    {
        convert<color_traits::format> cv;
        return cv( buffer[y * viewport.w + x] );
    }
};

template <typename CT>
struct color_access<CT, vector<4, simd::float4>>
{
    typedef typename CT::type output_color;
    typedef vector<4, simd::float4> input_color;

    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, input_color const& color, output_color* buffer)
    {
        using simd::mask4;
        using simd::store;

        input_color c = transpose(color);
        store( reinterpret_cast<float*>(&buffer[y * viewport.w + x]),             c.x, mask4(x < viewport.w && y < viewport.h) );
        store( reinterpret_cast<float*>(&buffer[y * viewport.w + (x + 1)]),       c.y, mask4((x + 1) < viewport.w && y < viewport.h) );
        store( reinterpret_cast<float*>(&buffer[(y + 1) * viewport.w + x]),       c.z, mask4(x < viewport.w && (y + 1) < viewport.h) );
        store( reinterpret_cast<float*>(&buffer[(y + 1) * viewport.w + (x + 1)]), c.w, mask4((x + 1) < viewport.w && (y + 1) < viewport.h) );
    }

    VSNRAY_CPU_FUNC
    static void blend(int x, int y, recti const& viewport, input_color const& color, output_color* buffer,
        simd::float4 sfactor, simd::float4 dfactor)
    {
        auto dst = get(x, y, viewport, buffer);
        dst = color * sfactor + dst * dfactor;
        store(x, y, viewport, dst, buffer);
    }

    VSNRAY_CPU_FUNC
    static input_color get(int x, int y, recti const& viewport, output_color* buffer)
    {
        output_color c00 = (x < viewport.w          && y < viewport.h)
            ? output_color(reinterpret_cast<float*>(&buffer[y * viewport.w + x])) : output_color();

        output_color c01 = ((x + 1) < viewport.w    && y < viewport.h)
            ? output_color(reinterpret_cast<float*>(&buffer[y * viewport.w + (x + 1)])) : output_color();

        output_color c10 = (x < viewport.w          && (y + 1) < viewport.h)
            ? output_color(reinterpret_cast<float*>(&buffer[(y + 1) * viewport.w + x])) : output_color();

        output_color c11 = ((x + 1) < viewport.w    && (y + 1) < viewport.h)
            ? output_color(reinterpret_cast<float*>(&buffer[(y + 1) * viewport.w + (x + 1)])) : output_color();

        return simd::pack(c00, c01, c10, c11);
    }
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
template <typename CT>
struct color_access<CT, vector<4, simd::float8>>
{
    typedef typename CT::type output_color;
    typedef vector<4, simd::float8> input_color;

    VSNRAY_CPU_FUNC
    static void store(int x, int y, recti const& viewport, input_color const& color, output_color* buffer)
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

        for (auto row = 0; row < packet_size<simd::float8>::h; ++row)
        {
            for (auto col = 0; col < packet_size<simd::float8>::w; ++col)
            {
                if (x + col < viewport.w && y + row < viewport.h)
                {
                    auto idx = row * 4 + col;
                    buffer[(y + row) * viewport.w + (x + col)] = output_color(r[idx], g[idx], b[idx], a[idx]);
                }
            }
        }
    }
};
#endif


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
#ifdef __CUDA_ARCH__
    return clock64();
#else
    auto t = std::chrono::high_resolution_clock::now();
    return t.time_since_epoch().count();
#endif
}

template <template <typename> class S, typename T>
class sampler_gen;

template <template <typename> class S>
class sampler_gen<S, float>
{
public:

    VSNRAY_FUNC sampler_gen() {}
    VSNRAY_FUNC sampler_gen(unsigned seed) : seed_(seed) {}

    template <typename V>
    VSNRAY_FUNC
    inline S<float> operator()(unsigned x, unsigned y, V const& viewport)
    {
        return S<float>( hash(seed_ + y * static_cast<unsigned>(viewport.w) + x) );
    }

private:

    unsigned seed_ = 0;

};

template <template <typename> class S>
class sampler_gen<S, simd::float4>
{
public:

    sampler_gen() = default;
    sampler_gen(unsigned seed) : seed_(seed) {}

    template <typename V>
    VSNRAY_CPU_FUNC
    inline S<simd::float4> operator()(unsigned x, unsigned y, V const& viewport)
    {
        return S<simd::float4>(seed_);
    }

private:

    unsigned seed_ = 0;

};


//-------------------------------------------------------------------------------------------------
// primary ray generators
//

template <typename R, typename T, typename V>
VSNRAY_FUNC
inline R ray_gen(T x, T y, V const& viewport, vector<3, T> const& eye,
    vector<3, T> const& cam_u, vector<3, T> const& cam_v, vector<3, T> const& cam_w)
{
    using scalar_type = T;

    auto u = scalar_type(2.0) * (x + scalar_type(0.5)) / scalar_type(viewport.w) - scalar_type(1.0);
    auto v = scalar_type(2.0) * (y + scalar_type(0.5)) / scalar_type(viewport.h) - scalar_type(1.0);

    R r;
    r.ori = eye;
    r.dir = normalize(cam_u * u + cam_v * v + cam_w);
    return r;
}

template <typename R, typename Mat4, typename V>
VSNRAY_FUNC
inline R ray_gen(typename R::scalar_type x, typename R::scalar_type y, V const& viewport,
    Mat4 const& inv_view_matrix, Mat4 const& inv_proj_matrix)
{
    typedef typename R::scalar_type scalar_type;
    typedef vector<4, scalar_type>  vec4_type;

    auto u = scalar_type(2.0) * (x + scalar_type(0.5)) / scalar_type(viewport.w) - scalar_type(1.0);
    auto v = scalar_type(2.0) * (y + scalar_type(0.5)) / scalar_type(viewport.h) - scalar_type(1.0);

    auto o = inv_view_matrix * ( inv_proj_matrix * vec4_type(u, v, -1,  1) );
    auto d = inv_view_matrix * ( inv_proj_matrix * vec4_type(u, v,  1,  1) );

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
// Simple uniform pixel sampler
//

template <typename R, typename CT, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::uniform_type, Args const&... args)
{
    VSNRAY_UNUSED(frame);

    typedef typename R::scalar_type     scalar_type;
    typedef vector<4, scalar_type>      vec4_type;

    auto r = uniform_ray_gen<R>(x, y, viewport, args...);
    auto color = kernel(r);
    color_access<CT, vec4_type>::store(x, y, viewport, color, color_buffer);
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, sampler is passed to kernel
//

template <typename R, typename CT, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::jittered_type, Args const&... args)
{
    VSNRAY_UNUSED(frame);

    typedef typename R::scalar_type     scalar_type;
    typedef C                           color_type;
    typedef vector<4, scalar_type>      vec4_type;

    auto gen    = sampler_gen<sampler, scalar_type>();
    auto s      = gen(x, y, viewport);
    auto r = jittered_ray_gen<R>(x, y, s, viewport, args...);
    auto color = kernel(r, s);
    color_access<CT, vec4_type>::store(x, y, viewport, color, color_buffer);
}


//-------------------------------------------------------------------------------------------------
// Jittered pixel sampler, result is blended on top of color buffer, sampler is passed to kernel
//

template <typename R, typename CT, typename V, typename C, typename K, typename ...Args>
VSNRAY_FUNC
inline void sample_pixel(unsigned x, unsigned y, unsigned frame, V const& viewport,
    C* color_buffer, K kernel, pixel_sampler::jittered_blend_type, Args const&... args)
{
    typedef typename R::scalar_type     scalar_type;
    typedef C                           color_type;
    typedef vector<4, scalar_type>      vec4_type;

    auto gen    = sampler_gen<sampler, scalar_type>(tic());
    auto s      = gen(x, y, viewport);
    auto r = jittered_ray_gen<R>(x, y, s, viewport, args...);
    auto color = kernel(r, s);
    auto alpha = scalar_type(1.0) / scalar_type(frame);
    if (frame <= 1)
    {//TODO: clear method in render target?
        color_access<CT, vec4_type>::store(x, y, viewport, vec4_type(0.0, 0.0, 0.0, 0.0), color_buffer);
    }
    color_access<CT, vec4_type>::blend(x, y, viewport, color, color_buffer,
        alpha, scalar_type(1.0) - alpha);
}


//-------------------------------------------------------------------------------------------------
// TODO: blend several samples at once
//

template <typename R, typename CT, typename Mat4, typename V, typename C, typename K>
VSNRAY_FUNC
inline void sample_pixel_XXX(unsigned x, unsigned y, Mat4 inv_view_matrix, Mat4 inv_proj_matrix,
    V viewport, C* color_buffer, K kernel)
{
    typedef typename R::scalar_type     scalar_type;
    typedef C                           color_type;
    typedef vector<4, scalar_type>      vec4_type;

    auto gen   = sampler_gen<sampler, scalar_type>();
    auto s     = gen(x, y, viewport);
    size_t spp = 8;
    for (size_t i = 0; i <= spp; ++i)
    {
        auto r     = jittered_ray_gen<R>(x, y, inv_view_matrix, inv_proj_matrix, viewport, s);
        auto src   = kernel(r, s);
        auto dst   = i == 0 ? vec4_type(0.0, 0.0, 0.0, 1.0) : color_access<CT, vec4_type>::get(x, y, viewport, color_buffer);
        auto alpha = 1.0f - static_cast<float>(i) / spp;
        dst = dst * scalar_type(1 - alpha) + src * scalar_type(alpha);
        color_access<CT, vec4_type>::store(x, y, viewport, dst, color_buffer);
    }
}

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_SCHED_COMMON_H


