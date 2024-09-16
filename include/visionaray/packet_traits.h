// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PACKET_TRAITS_H
#define VSNRAY_PACKET_TRAITS_H 1

#include "detail/macros.h"
#include "math/simd/simd.h"
//#include "math/simd/avx.h"
//#include "math/simd/avx512.h"
//#include "math/simd/builtin.h"
//#include "math/simd/neon.h"
//#include "math/simd/sse.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Traits to determine size and layout of ray packets
//
// Traits have default implementations where packets contain only a single ray
//
//  - packet_size (w,h)
//      determine width and height of a ray packet
//
//  - expand_pixel (x,y)
//      given a pixel position (x,y), returns simd vectors with pixel positions
//      expanded in the respective direction
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Type traits to determine packet size
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

template <>
struct packet_size<simd::float8>
{
    enum { w = 4, h = 2 };
};

template <>
struct packet_size<simd::float16>
{
    enum { w = 4, h = 4 };
};


//-------------------------------------------------------------------------------------------------
// Type traits to expand a pixel according to a ray packet type
//

template <typename T>
struct expand_pixel
{
    VSNRAY_FUNC inline constexpr T x(int x) { return T(x); }
    VSNRAY_FUNC inline constexpr T y(int y) { return T(y); }
};

template <>
struct expand_pixel<simd::float4>
{
    VSNRAY_FUNC inline simd::float4 x(int x) { return simd::float4(x, x + 1, x, x + 1); }
    VSNRAY_FUNC inline simd::float4 y(int y) { return simd::float4(y, y, y + 1, y + 1); }
};

template <>
struct expand_pixel<simd::float8>
{
    VSNRAY_FUNC inline simd::float8 x(int x)
    {
        return simd::float8(x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3);
    }

    VSNRAY_FUNC inline simd::float8 y(int y)
    {
        return simd::float8(y, y, y, y, y + 1, y + 1, y + 1, y + 1);
    }
};

template <>
struct expand_pixel<simd::float16>
{
    VSNRAY_FUNC inline simd::float16 x(int x)
    {
        return simd::float16(
                x, x + 1, x + 2, x + 3,
                x, x + 1, x + 2, x + 3,
                x, x + 1, x + 2, x + 3,
                x, x + 1, x + 2, x + 3
                );
    }

    VSNRAY_FUNC inline simd::float16 y(int y)
    {
        return simd::float16(
                y,     y,     y,     y,
                y + 1, y + 1, y + 1, y + 1,
                y + 2, y + 2, y + 2, y + 2,
                y + 3, y + 3, y + 3, y + 3
                );
    }
};

} // visionaray

#endif // VSNRAY_PACKET_TRAITS_H
