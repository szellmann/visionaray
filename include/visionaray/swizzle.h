// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SWIZZLE_H
#define VSNRAY_SWIZZLE_H 1

#include <cstddef>

#include "math/unorm.h"
#include "math/vector.h"
#include "pixel_format.h"

namespace visionaray
{

enum swizzle_hint
{
    PremultiplyAlpha,
    TruncateAlpha,
    AlphaIsZero,
    AlphaIsOne
};


namespace detail
{

//-------------------------------------------------------------------------------------------------
// Swizzle into 2nd data array
//

inline void swizzle_DEPTH24_STENCIL8_to_DEPTH32F(
        float*          dst,
        unsigned const* src,
        size_t          len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        unsigned depth = src[i] >> 8;
        dst[i] = depth / 16777215.0f;
    }
}

inline void swizzle_RGB16UI_to_RGB8(
        vector<3, unorm< 8>>*       dst,
        vector<3, unorm<16>> const* src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<3, unorm<8>>(
            static_cast<float>(rgb.x),
            static_cast<float>(rgb.y),
            static_cast<float>(rgb.z)
            );
    }
}

inline void swizzle_RGB32F_to_RGB8(
        vector<3, unorm<8>>*        dst,
        vector<3, float> const*     src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<3, unorm<8>>( rgb.x, rgb.y, rgb.z );
    }
}

inline void swizzle_RGBA16UI_to_RGBA8(
        vector<4, unorm< 8>>*       dst,
        vector<4, unorm<16>> const* src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgba = src[i];
        dst[i] = vector<4, unorm<8>>(
            static_cast<float>(rgba.x),
            static_cast<float>(rgba.y),
            static_cast<float>(rgba.z),
            static_cast<float>(rgba.w)
            );
    }
}

inline void swizzle_RGBA32F_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<4, float> const*     src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgba = src[i];
        dst[i] = vector<4, unorm<8>>( rgba.x, rgba.y, rgba.z, rgba.w );
    }
}

template <typename T, typename U>
inline void swizzle_RGBA_to_RGB(
        vector<3, T>*       dst,
        vector<4, U> const* src,
        size_t              len,
        swizzle_hint        hint
        )
{
    if (hint == PremultiplyAlpha)
    {
        for (size_t i = 0; i < len; ++i)
        {
            auto rgba = src[i];
            float alpha = static_cast<float>(rgba.w);
            dst[i] = vector<3, T>(
                    static_cast<float>(rgba.x) * alpha,
                    static_cast<float>(rgba.y) * alpha,
                    static_cast<float>(rgba.z) * alpha
                    );
        }
    }
    else if (hint == TruncateAlpha)
    {
        for (size_t i = 0; i < len; ++i)
        {
            auto rgba = src[i];
            dst[i] = vector<3, T>( rgba.x, rgba.y, rgba.z );
        }
    }
}

// Cast between unorm types with different bit depth
template <unsigned BitsDst, unsigned BitsSrc>
inline void swizzle_RGBA_to_RGB_cast_unorm(
        vector<3, unorm<BitsDst>>*          dst,
        vector<4, unorm<BitsSrc>> const*    src,
        size_t                              len,
        swizzle_hint                        hint
        )
{
    if (hint == PremultiplyAlpha)
    {
        for (size_t i = 0; i < len; ++i)
        {
            auto rgba = src[i];
            float alpha = static_cast<float>(rgba.w);
            dst[i] = vector<3, unorm<BitsDst>>(
                    static_cast<float>(rgba.x) * alpha,
                    static_cast<float>(rgba.y) * alpha,
                    static_cast<float>(rgba.z) * alpha
                    );
        }
    }
    else if (hint == TruncateAlpha)
    {
        for (size_t i = 0; i < len; ++i)
        {
            auto rgba = src[i];
            dst[i] = vector<3, unorm<BitsDst>>(
                    static_cast<float>(rgba.x),
                    static_cast<float>(rgba.y),
                    static_cast<float>(rgba.z)
                    );
        }
    }
}

// Cast between unorm types with different bit depth
template <unsigned BitsDst, unsigned BitsSrc>
inline void swizzle_RGB_to_RGBA_cast_unorm(
        vector<4, unorm<BitsDst>>*          dst,
        vector<3, unorm<BitsSrc>> const*    src,
        size_t                              len,
        swizzle_hint                        hint
        )
{
    float a = hint == AlphaIsZero ? 0.0f : 1.0f;
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<4, unorm<BitsDst>>(
                static_cast<float>(rgb.x),
                static_cast<float>(rgb.y),
                static_cast<float>(rgb.z),
                a
                );
    }
}

inline void swizzle_R8_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        unorm<8> const*             src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto r = src[i];
        dst[i] = vector<4, unorm<8>>( r, 0U, 0U, a );
    }
}

inline void swizzle_RG8_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<2, unorm<8>> const*  src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto rg = src[i];
        dst[i] = vector<4, unorm<8>>( rg.x, rg.y, 0U, a );
    }
}

inline void swizzle_RGB8_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<3, unorm<8>> const*  src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<4, unorm<8>>( rgb.x, rgb.y, rgb.z, a );
    }
}

inline void swizzle_R32F_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        float const*                src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto r = src[i];
        dst[i] = vector<4, unorm<8>>( r, 0U, 0U, a );
    }
}

inline void swizzle_RG32F_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<2, float> const*     src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto r = src[i].x;
        auto g = src[i].y;
        dst[i] = vector<4, unorm<8>>( r, g, 0U, a );
    }
}

inline void swizzle_RGB8_to_RGBA32F(
        vector<4, float>*           dst,
        vector<3, unorm<8>> const*  src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    unsigned char a = hint == AlphaIsZero ? 0U : 255U;
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        vector<4, unorm<8>> temp( rgb.x, rgb.y, rgb.z, a );
        dst[i] = vector<4, float>(temp);
    }
}

inline void swizzle_RGBA8_to_RGBA32F(
        vector<4, float>*           dst,
        vector<4, unorm<8>> const*  src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgba = src[i];
        dst[i] = vector<4, float>(rgba);
    }
}

inline void swizzle_RGB32F_to_RGBA32F(
        vector<4, float>*           dst,
        vector<3, float> const*     src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    float a = hint == AlphaIsZero ? 0.0f : 1.0f;
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<4, float>( rgb.x, rgb.y, rgb.z, a );
    }
}

inline void swizzle_RGB32F_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<3, float> const*     src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    float a = hint == AlphaIsZero ? 0.0f : 1.0f;
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<4, unorm<8>>( rgb.x, rgb.y, rgb.z, a );
    }
}


//-------------------------------------------------------------------------------------------------
// Swizzle in-place
//

inline void swizzle_BGR8_to_RGB8(vector<3, unorm<8>>* data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        auto bgr = data[i];
        data[i] = vector<3, unorm<8>>( bgr.z, bgr.y, bgr.x );
    }
}

inline void swizzle_BGRA8_to_RGBA8(vector<4, unorm<8>>* data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        auto bgra = data[i];
        data[i] = vector<4, unorm<8>>( bgra.z, bgra.y, bgra.x, bgra.w );
    }
}

inline void swizzle_RGB8_to_BGR8(vector<3, unorm<8>>* data, size_t len)
{
    // inverse
    return swizzle_BGR8_to_RGB8( data, len );
}

inline void swizzle_RGBA8_to_BGRA8(vector<4, unorm<8>>* data, size_t len)
{
    // inverse
    return swizzle_BGRA8_to_RGBA8( data, len );
}


//-------------------------------------------------------------------------------------------------
// Expand types for swizzling
//

// DEPTH24_STENCIL8 -> DEPTH32F

inline void swizzle_expand_types(
        float*          dst,
        pixel_format    format_dst,
        unsigned const* src,
        pixel_format    format_src,
        size_t          len
        )
{
    if (format_dst == PF_DEPTH32F && format_src == PF_DEPTH24_STENCIL8)
    {
        swizzle_DEPTH24_STENCIL8_to_DEPTH32F( dst, src, len );
    }
}

// RGBA8 -> RGB8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<4, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_RGBA8)
    {
        swizzle_RGBA_to_RGB( dst, src, len, hint );
    }
}

// RGBAUI16 -> RGB8, 16-bit type is unorm<16>, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm< 8>>*       dst,
        pixel_format                format_dst,
        vector<4, unorm<16>> const* src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_RGBA16UI)
    {
        swizzle_RGBA_to_RGB_cast_unorm( dst, src, len, hint );
    }
}

// R8 -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        unorm<8> const*             src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_R8)
    {
        detail::swizzle_R8_to_RGBA8( dst, src, len, hint );
    }
}

// RG8 -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<2, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RG8)
    {
        detail::swizzle_RG8_to_RGBA8( dst, src, len, hint );
    }
}

// RGB8 -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<3, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGB8)
    {
        detail::swizzle_RGB8_to_RGBA8( dst, src, len, hint );
    }
}

// RGB8 -> RGBA32F

inline void swizzle_expand_types(
        vector<4, float>*           dst,
        pixel_format                format_dst,
        vector<3, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA32F && format_src == PF_RGB8)
    {
        detail::swizzle_RGB8_to_RGBA32F( dst, src, len, hint );
    }
}

// R32F -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        float const*                src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_R32F)
    {
        detail::swizzle_R32F_to_RGBA8( dst, src, len, hint );
    }
}

// RG32F -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<2, float> const*     src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RG32F)
    {
        detail::swizzle_RG32F_to_RGBA8( dst, src, len, hint );
    }
}

// RGBA8 -> RGBA32F

inline void swizzle_expand_types(
        vector<4, float>*           dst,
        pixel_format                format_dst,
        vector<4, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len
        )
{
    if (format_dst == PF_RGBA32F && format_src == PF_RGBA8)
    {
        detail::swizzle_RGBA8_to_RGBA32F( dst, src, len );
    }
}

// RGB32F -> RGBA32F

inline void swizzle_expand_types(
        vector<4, float>*           dst,
        pixel_format                format_dst,
        vector<3, float> const*     src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA32F && format_src == PF_RGB32F)
    {
        detail::swizzle_RGB32F_to_RGBA32F( dst, src, len, hint );
    }
}

// RGB32F -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<3, float> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGB32F)
    {
        detail::swizzle_RGB32F_to_RGBA8( dst, src, len, hint );
    }
}

// RGBA32F -> RGB8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<4, float> const*  src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_RGBA32F)
    {
        detail::swizzle_RGBA_to_RGB( dst, src, len, hint );
    }
}

// RGB16UI -> RGB8, 16-bit type is unorm<16>, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm< 8>>*       dst,
        pixel_format                format_dst,
        vector<3, unorm<16>> const* src,
        pixel_format                format_src,
        size_t                      len
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_RGB16UI)
    {
        detail::swizzle_RGB16UI_to_RGB8( dst, src, len );
    }
}

// RGB16UI -> RGBA8, 16-bit type is unorm<16>, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm< 8>>*       dst,
        pixel_format                format_dst,
        vector<3, unorm<16>> const* src,
        pixel_format                format_src,
        size_t                      len,
        swizzle_hint                hint
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGB16UI)
    {
        detail::swizzle_RGB_to_RGBA_cast_unorm( dst, src, len, hint );
    }
}

// RGB32F -> RGB8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm<8>>*    dst,
        pixel_format            format_dst,
        vector<3, float> const* src,
        pixel_format            format_src,
        size_t                  len
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_RGB32F)
    {
        detail::swizzle_RGB32F_to_RGB8( dst, src, len );
    }
}

// RGBA16UI -> RGBA8, 16-bit type is unorm<16>, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm< 8>>*       dst,
        pixel_format                format_dst,
        vector<4, unorm<16>> const* src,
        pixel_format                format_src,
        size_t                      len
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGBA16UI)
    {
        detail::swizzle_RGBA16UI_to_RGBA8( dst, src, len );
    }
}

// RGBA32F -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*    dst,
        pixel_format            format_dst,
        vector<4, float> const* src,
        pixel_format            format_src,
        size_t                  len
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGBA32F)
    {
        detail::swizzle_RGBA32F_to_RGBA8( dst, src, len );
    }
}

// RGB8 <-> BGR8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<3, unorm<8>>*    data,
        pixel_format            format_dst,
        pixel_format            format_src,
        size_t                  len
        )
{
    if (format_dst == PF_RGB8 && format_src == PF_BGR8)
    {
        detail::swizzle_BGR8_to_RGB8( data, len );
    }
    else if (format_dst == PF_BGR8 && format_src == PF_RGB8)
    {
        detail::swizzle_RGB8_to_BGR8( data, len );
    }
}

// RGBA8 <-> BGRA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*    data,
        pixel_format            format_dst,
        pixel_format            format_src,
        size_t                  len
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_BGRA8)
    {
        detail::swizzle_BGRA8_to_RGBA8( data, len );
    }
    else if (format_dst == PF_BGRA8 && format_src == PF_RGBA8)
    {
        detail::swizzle_RGBA8_to_BGRA8( data, len );
    }
}

} // detail


//-------------------------------------------------------------------------------------------------
// Dispatch function for swizzling with two arrays
//

template <typename T, typename U>
inline void swizzle(
        T*              dst,
        pixel_format    format_dst,
        U const*        src,
        pixel_format    format_src,
        size_t          len
        )
{
    detail::swizzle_expand_types( dst, format_dst, src, format_src, len );
}


//-------------------------------------------------------------------------------------------------
// Dispatch function with two arrays and a hint about how to handle alpha (etc.)
//

template <typename T, typename U>
inline void swizzle(
        T*              dst,
        pixel_format    format_dst,
        U const*        src,
        pixel_format    format_src,
        size_t          len,
        swizzle_hint    hint
        )
{
    detail::swizzle_expand_types( dst, format_dst, src, format_src, len, hint );
}


//-------------------------------------------------------------------------------------------------
// Dispatch function for in-place swizzling
//

template <typename T>
inline void swizzle(
        T*              data,
        pixel_format    format_dst,
        pixel_format    format_src,
        size_t          len
        )
{
    detail::swizzle_expand_types( data, format_dst, format_src, len );
}

} // visionaray

#endif // VSNRAY_SWIZZLE_H
