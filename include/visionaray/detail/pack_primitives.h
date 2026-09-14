// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PACK_PRIMITIVES_H
#define VSNRAY_DETAIL_PACK_PRIMITIVES_H 1

#include "../math/simd/simd.h"
#include "../math/triangle.h"
#include "../aligned_vector.h"

namespace visionaray::simd
{

//-------------------------------------------------------------------------------------------------
// Convenience functions to pack alignment vectors for different primitive types
//

template <int W, typename Vector>
inline Vector pack_primitives(Vector const& v, aligned_vector<bvh_multi_node<W>, 32>& nodes)
{
    return v;
}

template <int W, typename P>
inline aligned_vector<P, 4 * W> pack_primitives(aligned_vector<P> const& v, aligned_vector<bvh_multi_node<W>, 32>& nodes)
{
    aligned_vector<P, 4 * W> result(v.size());
    std::memcpy(result.data(), v.data(), sizeof(v[0]) * v.size());
    return result;
}

template <int W>
inline aligned_vector<basic_triangle<3, float_from_simd_width_t<W>, int_from_simd_width_t<W>>, 4 * W> pack_primitives(
    aligned_vector<basic_triangle<3, float, unsigned>> const& v,
    aligned_vector<bvh_multi_node<W>, 32>& nodes
    )
{
    using F = float_from_simd_width_t<W>;
    using I = int_from_simd_width_t<W>;
    aligned_vector<basic_triangle<3, F, I>, 4 * W> result;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            int64_t addr = nodes[i].children[j];
            if (addr < 0)
            {
                uint64_t first_prim, num_prims;
                bvh_multi_node<W>::decode_leaf(addr, first_prim, num_prims);

                num_prims = div_up((int)num_prims, W) * W;

                int first = first_prim;
                int last  = first + num_prims;

                uint64_t new_first_prim = result.size();

                for (int k = first; k < last; k += W)
                {
                    array<basic_triangle<3, float, unsigned>, W> arr;
                    for (int l = 0; l < W; ++l)
                    {
                        arr[l] = v[(k + l) % v.size()];
                    }
                    result.push_back(simd::pack(arr));
                }

                uint64_t new_num_prims = result.size() - new_first_prim;
                nodes[i].children[j] = bvh_multi_node<W>::encode_leaf(new_first_prim, new_num_prims);
            }
        }
    }

    return result;
}

} // visionaray::simd

#endif // VSNRAY_DETAIL_PACK_PRIMITIVES_H
