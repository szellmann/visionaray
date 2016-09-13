// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PIXEL_ACCESS_H
#define VSNRAY_DETAIL_PIXEL_ACCESS_H 1

#include <array>
#include <type_traits>

#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/render_target.h>
#include <visionaray/result_record.h>

#include "color_conversion.h"
#include "macros.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Store, get and blend pixel values (color and depth)
//

namespace pixel_access
{

// Store ------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Store an input color to an output color buffer, apply color conversion
//

template <pixel_format DF, pixel_format SF, typename InputColor, typename OutputColor>
VSNRAY_FUNC
inline void store(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        InputColor const&           color,
        OutputColor*                buffer
        )
{
    VSNRAY_UNUSED(height);

    convert(
        pixel_format_constant<DF>{},
        pixel_format_constant<SF>{},
        buffer[y * width + x],
        color
        );
}

//-------------------------------------------------------------------------------------------------
// Store SIMD rgb color to RGB8 render target, apply conversion
// OutputColor must be rgb
// TODO: consolidate all RGB copies
// TODO: consolidate with rgba version
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGB8>      /* dst format */,
        pixel_format_constant<PF_RGB32F>    /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<3, FloatT> const&            color,
        OutputColor*                        buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

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
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                convert(
                    pixel_format_constant<PF_RGB8>{},
                    pixel_format_constant<PF_RGB32F>{},
                    buffer[(y + row) * width + (x + col)],
                    vec3(r[idx], g[idx], b[idx])
                    );
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Store SIMD rgb color to RGB32F render target, apply conversion
// OutputColor must be rgb
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGB32F>    /* dst format */,
        pixel_format_constant<PF_RGB32F>    /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<3, FloatT> const&            color,
        OutputColor*                        buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

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
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                convert(
                    pixel_format_constant<PF_RGB32F>{},
                    pixel_format_constant<PF_RGB32F>{},
                    buffer[(y + row) * width + (x + col)],
                    vec3(r[idx], g[idx], b[idx])
                    );
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Store SIMD rgba color to RGB32F render target, apply conversion
// OutputColor must be rgb
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGB32F>    /* dst format */,
        pixel_format_constant<PF_RGBA32F>   /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, FloatT> const&            color,
        OutputColor*                        buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

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
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                convert(
                    pixel_format_constant<PF_RGB32F>{},
                    pixel_format_constant<PF_RGBA32F>{},
                    buffer[(y + row) * width + (x + col)],
                    vec4(r[idx], g[idx], b[idx], a[idx])
                    );
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Store SIMD rgba color to RGBA8 render target, apply conversion
// OutputColor must be rgba
// TODO: consolidate all rgba copies
// TODO: consolidate with rgb version
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGBA8>     /* dst format */,
        pixel_format_constant<PF_RGBA32F>   /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, FloatT> const&            color,
        OutputColor*                        buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

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
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                convert(
                    pixel_format_constant<PF_RGBA8>{},
                    pixel_format_constant<PF_RGBA32F>{},
                    buffer[(y + row) * width + (x + col)],
                    vec4(r[idx], g[idx], b[idx], a[idx])
                    );
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Store SIMD rgba color to RGBA32F render target, apply conversion
// OutputColor must be rgba
//

template <
    typename OutputColor,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGBA32F>   /* dst format */,
        pixel_format_constant<PF_RGBA32F>   /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, FloatT> const&            color,
        OutputColor*                        buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

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
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                convert(
                    pixel_format_constant<PF_RGBA32F>{},
                    pixel_format_constant<PF_RGBA32F>{},
                    buffer[(y + row) * width + (x + col)],
                    vec4(r[idx], g[idx], b[idx], a[idx])
                    );
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Store SSE rgba color to RGBA32F color buffer, no conversion necessary
// Special treatment, can convert from SoA to AoS using transpose
//

VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_RGBA32F>   /* dst format */,
        pixel_format_constant<PF_RGBA32F>   /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, simd::float4> const&      color,
        vector<4, float>*                   buffer
        )
{
    using simd::store;

    auto c = transpose(color);

    if ( x      < width &&  y      < height) store( buffer[ y      * width +  x     ].data(), c.x);
    if ((x + 1) < width &&  y      < height) store( buffer[ y      * width + (x + 1)].data(), c.y);
    if ( x      < width && (y + 1) < height) store( buffer[(y + 1) * width +  x     ].data(), c.z);
    if ((x + 1) < width && (y + 1) < height) store( buffer[(y + 1) * width + (x + 1)].data(), c.w);
}

//-------------------------------------------------------------------------------------------------
// Store single SIMD channel to 32-bit FP buffer, no conversion
// TODO: consolidate various overloads
//

template <
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_DEPTH32F>  /* dst format */,
        pixel_format_constant<PF_DEPTH32F>  /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        FloatT const&                       value,
        float*                              buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

    float_array v;

    simd::store(v, value);

    auto w = packet_size<FloatT>::w;
    auto h = packet_size<FloatT>::h;

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                convert(
                    pixel_format_constant<PF_DEPTH32F>{},
                    pixel_format_constant<PF_DEPTH32F>{},
                    buffer[(y + row) * width + (x + col)],
                    v[row * w + col]
                    );
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Store single SIMD channel to 32-bit FP buffer, no conversion
// TODO: consolidate various overloads
//

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* dst format */,
        pixel_format_constant<PF_DEPTH32F>          /* src format */,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        T const&                                    value,
        unsigned*                                   buffer
        )
{
    using float_array = simd::aligned_array_t<T>;

    float_array v;

    simd::store(v, value);

    auto w = packet_size<T>::w;
    auto h = packet_size<T>::h;

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                convert(
                    pixel_format_constant<PF_DEPTH24_STENCIL8>{},
                    pixel_format_constant<PF_DEPTH32F>{},
                    buffer[(y + row) * width + (x + col)],
                    v[row * w + col]
                    );
            }
        }
    }
}

template <
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
VSNRAY_CPU_FUNC
inline void store(
        pixel_format_constant<PF_R32F>  /* dst format */,
        pixel_format_constant<PF_R32F>  /* src format */,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        FloatT const&                   value,
        float*                          buffer
        )
{
    using float_array = simd::aligned_array_t<FloatT>;

    float_array v;

    simd::store(v, value);

    auto w = packet_size<FloatT>::w;
    auto h = packet_size<FloatT>::h;

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                convert(
                    pixel_format_constant<PF_R32F>{},
                    pixel_format_constant<PF_R32F>{},
                    buffer[(y + row) * width + (x + col)],
                    v[row * w + col]
                    );
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Store color from result record to output color buffer
//

template <pixel_format DF, pixel_format SF, typename T, typename OutputColor>
VSNRAY_FUNC
inline void store(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        result_record<T> const&     rr,
        OutputColor*                buffer
        )
{
    store(
        pixel_format_constant<DF>{},
        pixel_format_constant<SF>{},
        x,
        y,
        width,
        height,
        rr.color,
        buffer
        );
}

//-------------------------------------------------------------------------------------------------
// Store color and depth from result record to output buffers
//

template <
    pixel_format DFC,
    pixel_format SFC,
    pixel_format DFD,
    pixel_format SFD,
    typename T,
    typename Color,
    typename Depth
    >
VSNRAY_FUNC
inline void store(
        pixel_format_constant<DFC>  /* dst format color */,
        pixel_format_constant<SFC>  /* src format color */,
        pixel_format_constant<DFD>  /* dst format depth */,
        pixel_format_constant<SFD>  /* src format depth */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        result_record<T> const&     rr,
        Color*                      color_buffer,
        Depth*                      depth_buffer
        )
{
    // Store color
    store(
        pixel_format_constant<DFC>{},
        pixel_format_constant<SFC>{},
        x,
        y,
        width,
        height,
        rr.color,
        color_buffer
        );

    // Store depth
    store(
        pixel_format_constant<DFD>{},
        pixel_format_constant<SFD>{},
        x,
        y,
        width,
        height,
        rr.depth,
        depth_buffer
        );
}


// Get -------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Get a color from an output color buffer, apply conversion
//

template <pixel_format DF, pixel_format SF, typename InputColor, typename OutputColor>
VSNRAY_FUNC
inline void get(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        InputColor&                 color,
        OutputColor const*          buffer
        )
{
    VSNRAY_UNUSED(height);

    convert(
        pixel_format_constant<DF>{},
        pixel_format_constant<SF>{},
        color,
        buffer[y * width + x]
        );
}

//-------------------------------------------------------------------------------------------------
// Get SSE rgba color from RGB32F color buffer, let alpha = 1.0
//

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_RGBA32F>   /* dst format */,
        pixel_format_constant<PF_RGB32F>    /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, simd::float4>&            color,
        vector<3, T> const*                 buffer
        )
{
    using OutputColor = vector<3, T>;

    std::array<OutputColor, 4> out;

    out[0] = ( x      < width &&  y      < height) ? buffer[ y      * width +  x     ] : OutputColor();
    out[1] = ((x + 1) < width &&  y      < height) ? buffer[ y      * width + (x + 1)] : OutputColor();
    out[2] = ( x      < width && (y + 1) < height) ? buffer[(y + 1) * width +  x     ] : OutputColor();
    out[3] = ((x + 1) < width && (y + 1) < height) ? buffer[(y + 1) * width + (x + 1)] : OutputColor();

    color = vector<4, simd::float4>(simd::pack(out), simd::float4(1.0f));
}

//-------------------------------------------------------------------------------------------------
// Get SSE rgba color from RGBA32F color buffer, no conversion necessary
//

template <typename OutputColor>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_RGBA32F>   /* dst format */,
        pixel_format_constant<PF_RGBA32F>   /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        vector<4, simd::float4>&            color,
        OutputColor const*                  buffer
        )
{
    std::array<OutputColor, 4> out;

    out[0] = ( x      < width &&  y      < height) ? buffer[ y      * width +  x     ] : OutputColor();
    out[1] = ((x + 1) < width &&  y      < height) ? buffer[ y      * width + (x + 1)] : OutputColor();
    out[2] = ( x      < width && (y + 1) < height) ? buffer[(y + 1) * width +  x     ] : OutputColor();
    out[3] = ((x + 1) < width && (y + 1) < height) ? buffer[(y + 1) * width + (x + 1)] : OutputColor();

    color = simd::pack(out);
}

//-------------------------------------------------------------------------------------------------
// Get SSE simd vector from scalar buffer
//

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_R32F>  /* dst format */,
        pixel_format_constant<PF_R32F>  /* src format */,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        simd::float4&                   result,
        T const*                        buffer
        )
{
    VSNRAY_ALIGN(16) float out[4];

    out[0] = ( x      < width &&  y      < height) ? buffer[ y      * width +  x     ] : T();
    out[1] = ((x + 1) < width &&  y      < height) ? buffer[ y      * width + (x + 1)] : T();
    out[2] = ( x      < width && (y + 1) < height) ? buffer[(y + 1) * width +  x     ] : T();
    out[3] = ((x + 1) < width && (y + 1) < height) ? buffer[(y + 1) * width + (x + 1)] : T();

    result = simd::float4(out);
}

// TODO: merge w/ overload(s) from above

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_DEPTH32F>  /* dst format */,
        pixel_format_constant<PF_DEPTH32F>  /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        simd::float4&                       result,
        T const*                            buffer
        )
{
    VSNRAY_ALIGN(16) float out[4];

    out[0] = ( x      < width &&  y      < height) ? buffer[ y      * width +  x     ] : T();
    out[1] = ((x + 1) < width &&  y      < height) ? buffer[ y      * width + (x + 1)] : T();
    out[2] = ( x      < width && (y + 1) < height) ? buffer[(y + 1) * width +  x     ] : T();
    out[3] = ((x + 1) < width && (y + 1) < height) ? buffer[(y + 1) * width + (x + 1)] : T();

    result = simd::float4(out);
}

// TODO: merge w/ overload(s) from above

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* dst format */,
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* src format */,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        simd::int4&                                 result,
        T const*                                    buffer
        )
{
    VSNRAY_ALIGN(16) int out[4];

    out[0] = ( x      < width &&  y      < height) ? buffer[ y      * width +  x     ] : T();
    out[1] = ((x + 1) < width &&  y      < height) ? buffer[ y      * width + (x + 1)] : T();
    out[2] = ( x      < width && (y + 1) < height) ? buffer[(y + 1) * width +  x     ] : T();
    out[3] = ((x + 1) < width && (y + 1) < height) ? buffer[(y + 1) * width + (x + 1)] : T();

    result = simd::int4(out);
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Get AVX rgba color from output color buffer, apply conversion
// OutputColor must be rgba (TODO: ensure through source pixel format!)
//

template <pixel_format DF, pixel_format SF, typename OutputColor>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        vector<4, simd::float8>&    color,
        OutputColor const*          buffer
        )
{
    const int w = packet_size<simd::float8>::w;
    const int h = packet_size<simd::float8>::h;

    vec4 v[w * h];

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                v[idx] = buffer[(y + row) * width + (x + col)];
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

//-------------------------------------------------------------------------------------------------
// Get AVX simd vector from scalar buffer
// TODO: consolidate w/ float4 version
//

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_R32F>  /* dst format */,
        pixel_format_constant<PF_R32F>  /* src format */,
        int                             x,
        int                             y,
        int                             width,
        int                             height,
        simd::float8&                   result,
        T const*                        buffer
        )
{
    const int w = packet_size<simd::float8>::w;
    const int h = packet_size<simd::float8>::h;

    VSNRAY_ALIGN(32) float out[8];

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                out[idx] = buffer[(y + row) * width + (x + col)];
            }
        }
    }

    result = simd::float8(out);
}

template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_DEPTH32F>  /* dst format */,
        pixel_format_constant<PF_DEPTH32F>  /* src format */,
        int                                 x,
        int                                 y,
        int                                 width,
        int                                 height,
        simd::float8&                       result,
        T const*                            buffer
        )
{
    const int w = packet_size<simd::float8>::w;
    const int h = packet_size<simd::float8>::h;

    VSNRAY_ALIGN(32) float out[8];

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                out[idx] = buffer[(y + row) * width + (x + col)];
            }
        }
    }

    result = simd::float8(out);
}

// TODO: merge w/ overload(s) from above
template <typename T>
VSNRAY_CPU_FUNC
inline void get(
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* dst format */,
        pixel_format_constant<PF_DEPTH24_STENCIL8>  /* src format */,
        int                                         x,
        int                                         y,
        int                                         width,
        int                                         height,
        simd::int8&                                 result,
        T const*                                    buffer
        )
{
    const int w = packet_size<simd::float8>::w;
    const int h = packet_size<simd::float8>::h;

    VSNRAY_ALIGN(32) int out[8];

    for (auto row = 0; row < h; ++row)
    {
        for (auto col = 0; col < w; ++col)
        {
            if (x + col < width && y + row < height)
            {
                auto idx = row * w + col;
                out[idx] = buffer[(y + row) * width + (x + col)];
            }
        }
    }

    result = simd::int8(out);
}

#endif


// Blend ------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Blend input and output colors, store in output buffer
//

template <pixel_format DF, pixel_format SF, typename InputColor, typename OutputColor, typename T>
VSNRAY_FUNC
inline void blend(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        InputColor const&           color,
        OutputColor*                buffer,
        T                           sfactor,
        T                           dfactor
        )
{
    InputColor dst;

    get(pixel_format_constant<SF>{}, pixel_format_constant<DF>{}, x, y, width, height, dst, buffer);

    dst = color * sfactor + dst * dfactor;

    store(pixel_format_constant<DF>{}, pixel_format_constant<SF>{}, x, y, width, height, dst, buffer);
}


//-------------------------------------------------------------------------------------------------
// Blend color from result record on top of output color buffer
//

template <pixel_format DF, pixel_format SF, typename S, typename OutputColor, typename T>
VSNRAY_FUNC
inline void blend(
        pixel_format_constant<DF>   /* dst format */,
        pixel_format_constant<SF>   /* src format */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        result_record<S> const&     rr,
        OutputColor*                color_buffer,
        T                           sfactor,
        T                           dfactor
        )
{
    blend(
        pixel_format_constant<DF>{},
        pixel_format_constant<SF>{},
        x,
        y,
        width,
        height,
        rr.color,
        color_buffer,
        sfactor,
        dfactor
        );
}

//-------------------------------------------------------------------------------------------------
// Blend color and depth from result record on top of output buffers
//

template <
    pixel_format DFC,
    pixel_format SFC,
    pixel_format DFD,
    pixel_format SFD,
    typename S,
    typename OutputColor,
    typename Depth,
    typename T
    >
VSNRAY_FUNC
inline void blend(
        pixel_format_constant<DFC>  /* dst format color */,
        pixel_format_constant<SFC>  /* src format color */,
        pixel_format_constant<DFD>  /* dst format depth */,
        pixel_format_constant<SFD>  /* src format depth */,
        int                         x,
        int                         y,
        int                         width,
        int                         height,
        result_record<S> const&     rr,
        OutputColor*                color_buffer,
        Depth*                      depth_buffer,
        T                           sfactor,
        T                           dfactor
        )
{
    // Blend color
    blend(
        pixel_format_constant<DFC>{},
        pixel_format_constant<SFC>{},
        x,
        y,
        width,
        height,
        rr.color,
        color_buffer,
        sfactor,
        dfactor
        );

    // Blend depth
    blend(
        pixel_format_constant<DFD>{},
        pixel_format_constant<SFD>{},
        x,
        y,
        width,
        height,
        rr.depth,
        depth_buffer,
        sfactor,
        dfactor
        );
}

} // pixel_access

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_PIXEL_ACCESS_H
