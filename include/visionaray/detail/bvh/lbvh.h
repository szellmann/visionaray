// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_LBVH_H
#define VSNRAY_DETAIL_BVH_LBVH_H 1

#include <algorithm>
#include <array>

#include <visionaray/aligned_vector.h>
#include <visionaray/morton.h>

#ifdef _WIN32
#include <intrin.h>
#endif

#include "build_top_down.h"

namespace visionaray
{
namespace detail
{

VSNRAY_FUNC
inline unsigned clz(unsigned val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __clz(val);
#elif defined(_WIN32)
    return __lzcnt(val);
#else
    return __builtin_clz(val);
#endif
}

} // detail

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
        unsigned morton_code;

        VSNRAY_FUNC
        bool operator<(prim_ref rhs) const
        {
            return morton_code < rhs.morton_code;
        }
    };

    using leaf_infos = std::array<leaf_info, 2>;

    aligned_vector<prim_ref> prim_refs;
    aligned_vector<aabb> prim_bounds;

    VSNRAY_FUNC
    int find_split(prim_ref const* refs, int first, int last) const
    {
        unsigned code_first = refs[first].morton_code;
        unsigned code_last  = refs[last - 1].morton_code;

        if (code_first == code_last)
        {
            return (first + last) / 2;
        }

        unsigned common_prefix = detail::clz(code_first ^ code_last);

        int result = first;
        int step = last - first;

        do
        {
            step = (step + 1) / 2;
            int next = result + step;

            if (next < last)
            {
                unsigned code = refs[next].morton_code;
                if (code_first == code || detail::clz(code_first ^ code) > common_prefix)
                {
                    result = next;
                }
            }
        }
        while (step > 1);

        return result;
    }

    template <typename Tree, typename P>
    Tree build(Tree /* */, P* primitives, size_t num_prims, int max_leaf_size = -1)
    {
        Tree tree(primitives, num_prims);

        detail::build_top_down(tree, *this, primitives, primitives + num_prims, max_leaf_size);

        return tree;
    }

    template <typename I>
    leaf_info init(I first, I last)
    {
        // Calculate bounding box for all primitives

        aabb scene_bounds;
        scene_bounds.invalidate();


        // Also calculate bounding box for all centroids

        aabb centroid_bounds;
        centroid_bounds.invalidate();


        prim_bounds.resize(last - first);
        aligned_vector<vec3> centroids(last - first);

        int i = 0;
        for (auto it = first; it != last; ++it, ++i)
        {
            prim_bounds[i] = get_bounds(*it);
            scene_bounds.insert(prim_bounds[i]);

            centroids[i] = prim_bounds[i].center();
            centroid_bounds.insert(centroids[i]);
        }


        // Calculate morton codes for centroids

        prim_refs.resize(last - first);

        for (int i = 0; i < last - first; ++i)
        {
            vec3 centroid = centroids[i];

            // Express centroid in [0..1] relative to bounding box
            centroid -= centroid_bounds.center();
            centroid = (centroid + centroid_bounds.size() * 0.5f) / centroid_bounds.size();

            // Quantize centroid to 10-bit
            centroid = min(max(centroid * 1024.0f, vec3(0.0f)), vec3(1023.0f));

            prim_refs[i].id = i;
            prim_refs[i].morton_code = morton_encode3D(
                    static_cast<int>(centroid.x),
                    static_cast<int>(centroid.y),
                    static_cast<int>(centroid.z)
                    );
        }

        std::stable_sort(prim_refs.begin(), prim_refs.end());

        return { 0, static_cast<int>(last - first), scene_bounds };
    }

    // Inserts primitive indices into INDICES.
    template <typename Indices>
    int insert_indices(Indices& indices, leaf_info const& leaf)
    {
        for (int i = leaf.first; i < leaf.last; ++i)
        {
            indices.push_back(prim_refs[i].id);
        }

        return leaf.last - leaf.first;
    }

    // Return true if the leaf should be split into two new leaves. In this case
    // sr.leaves contains the information of the left/right leaves and the
    // method returns true. If the leaf should not be split, returns false.
    template <typename Data>
    bool split(leaf_infos& childs, leaf_info const& leaf, Data const& /*data*/, int max_leaf_size)
    {
        if (leaf.last - leaf.first <= max_leaf_size)
        {
            return false;
        }

        int split = find_split(prim_refs.data(), leaf.first, leaf.last);

        childs[0].first = leaf.first;
        childs[0].last = split + 1;
        childs[0].prim_bounds.invalidate();
        for (int i = childs[0].first; i != childs[0].last; ++i)
        {
            childs[0].prim_bounds = combine(
                    childs[0].prim_bounds,
                    prim_bounds[prim_refs[i].id]
                    );
        }

        childs[1].first = split + 1;
        childs[1].last = leaf.last;
        childs[1].prim_bounds.invalidate();
        for (int i = childs[1].first; i != childs[1].last; ++i)
        {
            childs[1].prim_bounds = combine(
                    childs[1].prim_bounds,
                    prim_bounds[prim_refs[i].id]
                    );
        }

        return true;
    }


    // TODO:
    bool use_spatial_splits;
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_LBVH_H
