// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MORTON_H
#define VSNRAY_MORTON_H 1

#include "detail/macros.h"
#include "math/forward.h"
#include "math/vector.h"

namespace visionaray
{

VSNRAY_FUNC
inline unsigned morton_encode2D(unsigned x, unsigned y)
{
    auto separate_bits = [](unsigned n)
    {
        n &= 0x0000FFFF;
        n = (n ^ (n <<  8)) & 0x00FF00FF;
        n = (n ^ (n <<  4)) & 0x0F0F0F0F;
        n = (n ^ (n <<  2)) & 0x33333333;
        n = (n ^ (n <<  1)) & 0x55555555;
        return n;
    };

    return separate_bits(x) | (separate_bits(y) << 1);
}

VSNRAY_FUNC
inline unsigned morton_encode3D(unsigned x, unsigned y, unsigned z)
{
    auto separate_bits = [](unsigned n)
    {
        n &= 0x000003FF;
        n = (n ^ (n << 16)) & 0xFF0000FF;
        n = (n ^ (n <<  8)) & 0x0300F00F;
        n = (n ^ (n <<  4)) & 0x030C30C3;
        n = (n ^ (n <<  2)) & 0x09249249;
        return n;
    };  

    return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2); 
}

VSNRAY_FUNC
inline vec2ui morton_decode2D(unsigned index)
{
    auto compact_bits = [](unsigned n)
    {
        n &= 0x55555555;
        n = (n ^ (n >>  1)) & 0x33333333;
        n = (n ^ (n >>  2)) & 0x0F0F0F0F;
        n = (n ^ (n >>  4)) & 0x00FF00FF;
        n = (n ^ (n >>  8)) & 0x0000FFFF;
        return n;
    };

    return { compact_bits(index), compact_bits(index >> 1) };
}

VSNRAY_FUNC
inline vec3ui morton_decode3D(unsigned index)
{
    auto compact_bits = [](unsigned n)
    {
        n &= 0x09249249;
        n = (n ^ (n >>  2)) & 0x030C30C3;
        n = (n ^ (n >>  4)) & 0x0300F00F;
        n = (n ^ (n >>  8)) & 0xFF0000FF;
        n = (n ^ (n >> 16)) & 0x000003FF;
        return n;
    };  

    return { compact_bits(index), compact_bits(index >> 1), compact_bits(index >> 2) };
}

} // visionaray

#endif // VSNRAY_MORTON_H
