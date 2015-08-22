// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SWIZZLE_H
#define VSNRAY_SWIZZLE_H 1

#include <vector>

#include <visionaray/math/math.h>
#include <visionaray/pixel_format.h>

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

inline void swizzle_RGB8_to_RGBA8(
        vector<4, unorm<8>>*        dst,
        vector<3, unorm<8>> const*  src,
        size_t                      len
        )
{
    for (size_t i = 0; i < len; ++i)
    {
        auto rgb = src[i];
        dst[i] = vector<4, unorm<8>>( rgb.x, rgb.y, rgb.z, 255U );
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

// RGB8 -> RGBA8, 8-bit type is unorm<8>

inline void swizzle_expand_types(
        vector<4, unorm<8>>*        dst,
        pixel_format                format_dst,
        vector<3, unorm<8>> const*  src,
        pixel_format                format_src,
        size_t                      len
        )
{
    if (format_dst == PF_RGBA8 && format_src == PF_RGB8)
    {
        detail::swizzle_RGB8_to_RGBA8( dst, src, len );
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
// Dispatch function with two arrays and a hint about how to handle alpha
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
