// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_LBVH_H
#define VSNRAY_DETAIL_BVH_LBVH_H 1

#include <algorithm>
#include <array>

#include <visionaray/aligned_vector.h>
#include <visionaray/morton.h>

#include "/Users/stefan/visionaray/src/common/timer.h"
#include <visionaray/math/io.h>
#include <iostream>
#include <ostream>

namespace visionaray
{
namespace detail
{

inline unsigned count_leading_zeros(unsigned val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0 /*TODO: clz since Fermi*/
    return __clz(val);
#else
    return __builtin_clz(val);
#endif
}

struct lbvh_builder
{
    struct leaf_info
    {
        int first;
        int last;
        aabb prim_bounds;
    };

    struct prim_ref
    {
        int id;
        int morton_code;

        bool operator<(prim_ref rhs) const
        {
            return morton_code < rhs.morton_code;
        }
    };

    using leaf_infos = std::array<leaf_info, 2>;

    aligned_vector<prim_ref> prim_refs;
    aligned_vector<aabb> prim_bounds;

    VSNRAY_FUNC
    int find_split(int first, int last) const
    {
    }

    template <typename I>
    VSNRAY_FUNC
    leaf_info init(I first, I last)
    {
        timer t;

        // Calculate bounding box for all primitives
        // and centroids of primitive bounding boxes

        aabb scene_bounds;
        scene_bounds.invalidate();

        prim_bounds.resize(last - first);
        aligned_vector<vec3> centroids(last - first);

        int i = 0;
        for (auto it = first; it != last; ++it, ++i)
        {
            prim_bounds[i] = get_bounds(*it);
            scene_bounds.insert(prim_bounds[i]);
            centroids[i] = prim_bounds[i].center();
        }


        // Calculate morton codes for centroids

        prim_refs.resize(last - first);

        for (int i = 0; i < last - first; ++i)
        {
            vec3 centroid = centroids[i];

            // Express centroid in [0..1] relative to bounding box
            centroid = (centroid - scene_bounds.center()) / scene_bounds.size();

            // Quantize centroid to 10-bit
            centroid = min(max(centroid * 1024.0f, vec3(0.0f)), vec3(1023.0f));

            prim_refs[i].id = i;
            prim_refs[i].morton_code = morton_encode3D(
                    static_cast<int>(centroid.x),
                    static_cast<int>(centroid.y),
                    static_cast<int>(centroid.z)
                    );
        }

        std::cout << t.elapsed() << '\n';

        t.reset();

        std::sort(prim_refs.begin(), prim_refs.end());

        std::cout << t.elapsed() << '\n';

        return { 0, static_cast<int>(last - first), scene_bounds };
    }

    // Inserts primitive indices into INDICES.
    template <typename Indices>
    int insert_indices(Indices& indices, leaf_info const& leaf)
    {
        int leaf_size = leaf.last - leaf.first;

        for (int i = 0; i < leaf_size; ++i)
        {
            indices.push_back(leaf.first + i);
        }

        return leaf_size;
    }

    // Return true if the leaf should be split into two new leaves. In this case
    // sr.leaves contains the information of the left/right leaves and the
    // method returns true. If the leaf should not be split, returns false.
    template <typename Data>
    bool split(leaf_infos& childs, leaf_info const& leaf, Data const& data, int max_leaf_size)
    {
        if (leaf.last - leaf.first == 1)
        {
            return false;
        }

        int split = find_split(leaf.first, leaf.last);

        childs[0].first = leaf.first;
        childs[0].last = split;
        childs[0].prim_bounds = combine(prim_bounds[leaf.first], prim_bounds[split - 1]);

        childs[1].first = split;
        childs[1].last = leaf.last;
        childs[1].prim_bounds = combine(prim_bounds[split], prim_bounds[leaf.last - 1]);

        return true;
    }


    // TODO:
    bool use_spatial_splits;
};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_BVH_LBVH_H
