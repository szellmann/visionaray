// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_LBVH_COMMON_H
#define VSNRAY_DETAIL_BVH_LBVH_COMMON_H 1

#include <visionaray/math/math.h>
#include <visionaray/morton.h>

namespace visionaray
{
namespace detail
{

struct CustomLess
{
    template <typename DataType>
    VSNRAY_GPU_FUNC bool operator()(DataType const& lhs, DataType const& rhs)
    {
        return lhs < rhs;
    }
};


VSNRAY_FUNC
inline unsigned clz(unsigned val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __clz(val);
#elif defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    return __clz(val);
#elif defined(_WIN32)
    return __lzcnt(val);
#else
    return __builtin_clz(val);
#endif
}


namespace lbvh
{

//-------------------------------------------------------------------------------------------------
// Map primitive to morton code
//

struct prim_ref
{
    int id;
    unsigned morton_code;

    VSNRAY_FUNC
    bool operator<(prim_ref rhs) const
    {
        return morton_code < rhs.morton_code;
    }
};


//-------------------------------------------------------------------------------------------------
// Find node range that an inner node overlaps
//

VSNRAY_FUNC
inline vec2i determine_range(prim_ref* refs, int num_prims, int i, int& split)
{
    auto delta = [&](int i, int j)
    {
        // Karras' delta(i,j) function
        // Denotes the length of the longest common
        // prefix between keys k_i and k_j

        // Cf. Figure 4: "for simplicity, we define that
        // delta(i,j) = -1 when j not in [0,n-1]"
        if (j < 0 || j >= num_prims)
        {
            return -1;
        }

        unsigned xord = refs[i].morton_code ^ refs[j].morton_code;
        if (xord == 0)
        {
            return static_cast<int>(clz((unsigned)i ^ (unsigned)j) + 32);
        }
        else
        {
            return static_cast<int>(clz(refs[i].morton_code ^ refs[j].morton_code));
        }
    };

    // Determine direction of the range (+1 or -1)
    int d = delta(i, i + 1) >= delta(i, i - 1) ? 1 : -1;

    // Compute upper bound for the length of the range
    int delta_min = delta(i, i - d);
    int l_max = 2;
    while (delta(i, i + l_max * d) > delta_min)
    {
        l_max *= 2;
    }

    // Find the other end using binary search
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1)
    {
        if (delta(i, i + (l + t) * d) > delta_min)
            l += t;
    }

    int j = i + l * d;

    // Find the split position using binary search
    int delta_node = delta(i, j);
    int s = 0;
    float divf = 2.f;
    int t = ceil(l / divf);
    for(; t >= 1; divf *= 2.f, t = ceil(l / divf))
    {
        if (delta(i, i + (s + t) * d) > delta_node)
            s += t;
    }

    split = i + s * d + min(d, 0);

    if (d == 1)
        return vec2i(i, j);
    else
        return vec2i(j, i);
}

} // lbvh
} // detail
} // visionaray

#endif // VSNRAY_DETAIL_BVH_LBVH_COMMON_H
