// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_STATISTICS_H
#define VSNRAY_DETAIL_BVH_STATISTICS_H 1

#include <type_traits>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Compute the SAH cost for a BVH
//
// cf. Karras, Aila (2013): Fast Parallel Construction of High-Quality Bounding Volume Hierarchies
//
// Parameters:
//
// [in] B
//      BVH tree
//
// [in] CI
//      Estimated costs to traverse an inner node
//
// [in] CL
//      Estimated costs to traverse a leaf node
//
// [in] CP
//      Estimated costs to intersect a primitive
//

template <
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type
    >
inline float sah_cost(BVH const& b, float ci = 1.2f, float cl = 0.0f, float cp = 1.0f)
{
    // Surface area of root
    float A_r = surface_area(b.node(0).get_bounds());

    // Summed surface area of all leaves
    float A_l = 0.0f;

    // Summed surface area of all inner nodes
    float A_i = 0.0f;

    // Summed surface area of all leaves times #primitives of all leaves
    float A_l_x_N_n = 0.0f;

    for (auto n : b.nodes())
    {
        if (is_leaf(n))
        {
            A_l += surface_area(n.get_bounds());
            A_l_x_N_n += surface_area(n.get_bounds()) * static_cast<float>(n.get_num_primitives());
        }
        else
        {
            A_i += surface_area(n.get_bounds());
        }
    }

    return ci * (A_i / A_r)
         + cl * (A_l / A_r)
         + cp * (A_l_x_N_n / A_r);
}


//-------------------------------------------------------------------------------------------------
// Recursively compute the SAH cost of a node
//
// Parameters:
//
// [in] B
//      BVH tree
//
// [in] N
//      BVH node
//
// [in] CI
//      Estimated costs to traverse an inner node
//
// [in] CP
//      Estimated costs to intersect a primitive
//

template <
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type
    >
inline float sah_cost(BVH const& b, bvh_node const& n, float ci = 1.2f, float cp = 1.0f)
{
    if (is_inner(n))
    {
        return ci * surface_area(n.get_bounds())
             + sah_cost(b, b.node(n.get_child(0)))
             + sah_cost(b, b.node(n.get_child(1)));
    }
    else
    {
        return cp * surface_area(n.get_bounds()) * static_cast<float>(n.get_num_primitives());
    }
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_STATISTICS_H
