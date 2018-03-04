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
inline int morton_encode3D(int x, int y, int z)
{
    auto separate_bits = [](int n)
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
inline vec3i morton_decode3D(int index)
{
    auto compact_bits = [](int n)
    {
        n &= 0x09249249;
        n = (n ^ (n >>  2)) & 0x030C30C3;
        n = (n ^ (n >>  4)) & 0x0300F00F;
        n = (n ^ (n >>  8)) & 0xFF0000FF;
        n = (n ^ (n >> 16)) & 0x000003FF;
        return n;
    };  

    vec3i result(compact_bits(index));
    result.y >>= 1;
    result.z >>= 2;
    return result;
}

} // visionaray

#endif // VSNRAY_MORTON_H
