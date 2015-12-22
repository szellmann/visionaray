// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PIXEL_ACCESS_H
#define VSNRAY_DETAIL_PIXEL_ACCESS_H 1

#include <visionaray/pixel_format.h>
#include <visionaray/render_target.h>

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
// Store, get and blend pixel values (color and depth)
//

namespace pixel_access
{

// Store ------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Store an input color to an output color buffer, apply color conversion
//

template <typename InputColor, typename OutputColor>
VSNRAY_FUNC
void store(int x, int y, recti const& viewport, InputColor const& color, OutputColor* buffer)
{
    convert(buffer[y * viewport.w + x], color);
}

//-------------------------------------------------------------------------------------------------
// Store SIMD rgb color, apply conversion
// OutputColor must be rgb
// TODO: consolidate with rgba version
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
void store(int x, int y, recti const& viewport, vector<3, FloatT> const& color, OutputColor* buffer)
{
    using float_array = typename simd::aligned_array<FloatT>::type;

    float_array r;
    float_array g;
    float_array b;

    using simd::store;

    store(r, color.x);
    store(g, color.y);
    store(b, color.z);

    auto w = packet_size<FloatT>::w;
    auto h = packet_size<FloatT>::h;

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < viewport.w && y + row < viewport.h)
            {
                auto idx = row * w + col;
                convert( buffer[(y + row) * viewport.w + (x + col)], vec3(r[idx], g[idx], b[idx]) );
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Store SIMD rgba color, apply conversion
// OutputColor must be rgba
// TODO: consolidate with rgb version
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
void store(int x, int y, recti const& viewport, vector<4, FloatT> const& color, OutputColor* buffer)
{
    using float_array = typename simd::aligned_array<FloatT>::type;

    float_array r;
    float_array g;
    float_array b;
    float_array a;

    using simd::store;

    store(r, color.x);
    store(g, color.y);
    store(b, color.z);
    store(a, color.w);

    auto w = packet_size<FloatT>::w;
    auto h = packet_size<FloatT>::h;

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

//-------------------------------------------------------------------------------------------------
// Store SSE rgba color to RGBA32F color buffer, no conversion necessary
// Special treatment, can convert from SoA to AoS using transpose
//

VSNRAY_CPU_FUNC
void store(int x, int y, recti const& viewport, vector<4, simd::float4> const& color, vector<4, float>* buffer)
{
    using simd::store;

    auto c = transpose(color);

    if ( x      < viewport.w &&  y      < viewport.h) store( buffer[ y      * viewport.w +  x     ].data(), c.x);
    if ((x + 1) < viewport.w &&  y      < viewport.h) store( buffer[ y      * viewport.w + (x + 1)].data(), c.y);
    if ( x      < viewport.w && (y + 1) < viewport.h) store( buffer[(y + 1) * viewport.w +  x     ].data(), c.z);
    if ((x + 1) < viewport.w && (y + 1) < viewport.h) store( buffer[(y + 1) * viewport.w + (x + 1)].data(), c.w);
}

//-------------------------------------------------------------------------------------------------
// Store single SIMD channel to 32-bit FP buffer, no conversion
// Can be used for color and depth
//

template <
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
void store(int x, int y, recti const& viewport, FloatT const& value, float* buffer)
{
    using float_array = typename simd::aligned_array<FloatT>::type;

    float_array v;

    simd::store(v, value);

    auto w = packet_size<FloatT>::w;
    auto h = packet_size<FloatT>::h;

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

//-------------------------------------------------------------------------------------------------
// Store color from result record to output color buffer
//

template <typename T, typename OutputColor>
VSNRAY_FUNC
void store(int x, int y, recti const& viewport, result_record<T> const& rr, OutputColor* buffer)
{
    store(x, y, viewport, rr.color, buffer);
}

//-------------------------------------------------------------------------------------------------
// Store color and depth from result record to output buffers
//

template <typename T, typename Color, typename Depth>
VSNRAY_FUNC
void store(
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


// Get -------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Get a color from an output color buffer, apply conversion
//

template <typename InputColor, typename OutputColor>
VSNRAY_FUNC
void get(int x, int y, recti const& viewport, InputColor& color, OutputColor const* buffer)
{
    convert(color, buffer[y * viewport.w + x]);
}

//-------------------------------------------------------------------------------------------------
// Get SSE rgba color from RGBA32F color buffer, no conversion necessary
//

template <typename OutputColor>
VSNRAY_CPU_FUNC
void get(int x, int y, recti const& viewport, vector<4, simd::float4>& color, OutputColor const* buffer)
{
    auto c00 = ( x      < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w +  x     ] : OutputColor();
    auto c01 = ((x + 1) < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w + (x + 1)] : OutputColor();
    auto c10 = ( x      < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w +  x     ] : OutputColor();
    auto c11 = ((x + 1) < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w + (x + 1)] : OutputColor();

    color = simd::pack(vec4(c00), vec4(c01), vec4(c10), vec4(c11));
}

//-------------------------------------------------------------------------------------------------
// Get SSE rgba color from RGB32F color buffer, let alpha = 1.0
//

template <typename T>
VSNRAY_CPU_FUNC
void get(int x, int y, recti const& viewport, vector<4, simd::float4>& color, vector<3, T> const* buffer)
{
    using OutputColor = vector<3, T>;

    auto c00 = ( x      < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w +  x     ] : OutputColor();
    auto c01 = ((x + 1) < viewport.w &&  y      < viewport.h) ? buffer[ y      * viewport.w + (x + 1)] : OutputColor();
    auto c10 = ( x      < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w +  x     ] : OutputColor();
    auto c11 = ((x + 1) < viewport.w && (y + 1) < viewport.h) ? buffer[(y + 1) * viewport.w + (x + 1)] : OutputColor();

    color = simd::pack(vec4(c00, 1.0f), vec4(c01, 1.0f), vec4(c10, 1.0f), vec4(c11, 1.0f));
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Get AVX rgba color from output color buffer, apply conversion
// OutputColor must be rgba
//

template <typename OutputColor>
VSNRAY_CPU_FUNC
void get(int x, int y, recti const& viewport, vector<4, simd::float8>& color, OutputColor const* buffer)
{
    const int w = packet_size<simd::float8>::w;
    const int h = packet_size<simd::float8>::h;

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


// Blend ------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Blend input and output colors, store in output buffer
//

template <typename InputColor, typename OutputColor, typename T>
VSNRAY_FUNC
void blend(int x, int y, recti const& viewport, InputColor const& color, OutputColor* buffer, T sfactor, T dfactor)
{
    InputColor dst;

    get(x, y, viewport, dst, buffer);

    dst = color * sfactor + dst * dfactor;

    store(x, y, viewport, dst, buffer);
}


//-------------------------------------------------------------------------------------------------
// Blend color from result record on top of output color buffer
//

template <typename S, typename OutputColor, typename T>
VSNRAY_FUNC
void blend(
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

//-------------------------------------------------------------------------------------------------
// Blend color and depth from result record on top of output buffers
//

template <typename S, typename OutputColor, typename Depth, typename T>
VSNRAY_FUNC
void blend(
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

} // pixel_access

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_PIXEL_ACCESS_H
