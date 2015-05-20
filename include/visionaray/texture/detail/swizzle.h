// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SWIZZLE_H
#define VSNRAY_TEXTURE_SWIZZLE_H

#include <vector>

#include <visionaray/math/math.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Swizzle into 2nd data array
//

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

inline void swizzle_BGRA8_to_RGBA8(vector<4, unorm<8>>* data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        auto bgra = data[i];
        data[i] = vector<4, unorm<8>>( bgra.z, bgra.y, bgra.x, bgra.w );
    }
}

} // detail


//-------------------------------------------------------------------------------------------------
// Dispatch function for swizzling with two arrays
//

template <typename T, typename U>
inline void swizzle(
        T*              dst,
        U const*        src,
        size_t          len,
        pixel_format    format_source,
        pixel_format    format_dest
        )
{
    if (format_source == PF_RGB8 && format_dest == PF_RGBA8)
    {
        detail::swizzle_RGB8_to_RGBA8( dst, src, len );
    }
}


//-------------------------------------------------------------------------------------------------
// Dispatch function for in-place swizzling
//

template <typename T>
inline void swizzle(
        T*              data,
        size_t          len,
        pixel_format    format_source,
        pixel_format    format_dest
        )
{
    if (format_source == PF_BGRA8 && format_dest == PF_RGBA8)
    {
        detail::swizzle_BGRA8_to_RGBA8( data, len );
    }
}

} // visionaray

#endif // VSNRAY_TEXTURE_SWIZZLE_H
