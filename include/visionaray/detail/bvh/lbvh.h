// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_LBVH_H
#define VSNRAY_DETAIL_BVH_LBVH_H 1

#include <algorithm>
#include <array>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

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

//-------------------------------------------------------------------------------------------------
// Helpers
//

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

#ifdef __CUDACC__

//-------------------------------------------------------------------------------------------------
// Stolen from https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
//

VSNRAY_GPU_FUNC
inline float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        {
            break;
        }
    }
    return __int_as_float(ret);
}

VSNRAY_GPU_FUNC
inline float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        {
            break;
        }
    }
    return __int_as_float(ret);
}

#endif


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

#ifdef __CUDACC__

//-------------------------------------------------------------------------------------------------
// Node data structure only used for construction w/ Karras' algorithm. Alignment is bad, but
// node has parent pointer!
//

struct node
{
    VSNRAY_GPU_FUNC node()
        : bbox(vec3(numeric_limits<float>::max()), vec3(-numeric_limits<float>::max()))
    {
    }

    aabb bbox;
    int left = -1;
    int right = -1;
    int parent = -1;
};


//-------------------------------------------------------------------------------------------------
// CUDA kernels
//

template <typename P>
static __global__ void compute_bounds_and_centroids(
        aabb*    prim_bounds,     // OUT: all primitive bounding boxes
        vec3*    centroids,       // OUT: all primitive centroids
        aabb*    scene_bounds,    // OUT: the scene bounding box
        aabb*    centroid_bounds, // OUT: the centroid bounding box
        P const* primitives,      // IN:  all primitives
        int      num_prims        // IN:  number of primitives
        )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_prims)
    {
        prim_bounds[index] = get_bounds(primitives[index]);
        //scene_bounds->insert(prim_bounds[index]); // TODO: atomic (necessary?)

        centroids[index] = prim_bounds[index].center();
        atomicMin(&centroid_bounds->min.x, centroids[index].x);
        atomicMin(&centroid_bounds->min.y, centroids[index].y);
        atomicMin(&centroid_bounds->min.z, centroids[index].z);
        atomicMax(&centroid_bounds->max.x, centroids[index].x);
        atomicMax(&centroid_bounds->max.y, centroids[index].y);
        atomicMax(&centroid_bounds->max.z, centroids[index].z);
    }
}

static __global__ void assign_morton_codes(
        prim_ref*   prim_refs,       // OUT: prim refs with morton codes
        vec3 const* centroids,       // IN:  all centroids
        aabb*       centroid_bounds, // IN:  the centroid bounding box
        int         num_prims        // IN:  number of primitives
        )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_prims)
    {
        vec3 centroid = centroids[index];

        // Express centroid in [0..1] relative to bounding box
        centroid -= centroid_bounds->center();
        centroid = (centroid + centroid_bounds->size() * 0.5f) / centroid_bounds->size();

        // Quantize centroid to 10-bit
        centroid = min(max(centroid * 1024.0f, vec3(0.0f)), vec3(1023.0f));

        prim_refs[index].id = index;
        prim_refs[index].morton_code = morton_encode3D(
                static_cast<int>(centroid.x),
                static_cast<int>(centroid.y),
                static_cast<int>(centroid.z)
                );
    }
}

static __global__ void build_hierarchy(
        node*     inner,     // OUT: all inner nodes with pointers assigned
        node*     leaves,    // OUT: all leaf nodes with parent pointers assigned
        prim_ref* prim_refs, // IN:  prim refs with morton codes
        int       num_prims  // IN:  number of primitives
        )
{
    int num_leaves = num_prims;
    int num_inner = num_leaves - 1;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_inner)
    {
        // NOTE: This is [first..last], not [first..last)!!
        int split = -1;
        vec2i range = determine_range(prim_refs, num_prims, index, split);
        int first = range.x;
        int last = range.y;

        int left = split;
        int right = split + 1;

        if (left == first)
        {
            // left child is leaf
            inner[index].left = num_inner + left;
            leaves[left].parent = index;
        }
        else
        {
            // left child is inner
            inner[index].left = left;
            inner[left].parent = index;
        }

        if (right == last)
        {
            // right child is leaf
            inner[index].right = num_inner + right;
            leaves[right].parent = index;
        }
        else
        {
            // right child is inner
            inner[index].right = right;
            inner[right].parent = index;
        }
    }
}

static __global__ void assign_node_bounds(
        node*     inner,        // IN:  all inner nodes
        node*     leaves,       // IN:  all leaf nodes
        aabb*     prim_bounds,  // IN:  all primitive bounding boxes
        prim_ref* prim_refs,    // IN:  all prim refs
        int       num_prims     // IN:  number of primitives
        )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int num_leaves = num_prims;

    if (index >= num_leaves)
    {
        return;
    }

    // Start with leaf
    leaves[index].bbox = prim_bounds[prim_refs[index].id];

    // Atomically combine child bounding boxes and update parents
    int next = leaves[index].parent;

    while (next >= 0)
    {
        atomicMin(&inner[next].bbox.min.x, leaves[index].bbox.min.x);
        atomicMin(&inner[next].bbox.min.y, leaves[index].bbox.min.y);
        atomicMin(&inner[next].bbox.min.z, leaves[index].bbox.min.z);
        atomicMax(&inner[next].bbox.max.x, leaves[index].bbox.max.x);
        atomicMax(&inner[next].bbox.max.y, leaves[index].bbox.max.y);
        atomicMax(&inner[next].bbox.max.z, leaves[index].bbox.max.z);

        if (inner[next].parent == -1)
        {
            break;
        }

        // Traverse up
        next = inner[next].parent;
    }
}

static __global__ void collapse(
        bvh_node* bvh_nodes,    // OUT: visionaray bvh nodes
        node*     inner,        // IN:  all inner nodes
        node*     leaves,       // IN:  all leaf nodes
        prim_ref* prim_refs,    // IN:  all prim refs
        int       num_prims     // IN:  number of primitives
        )
{
    // Inline function to determine index into bvh_nodes
    // array (stores the two children next to each other,
    // while the input node arrays don't)
    auto bvh_node_index = [&](int current, int parent)
    {
        int result = 1; // step over root

        if (inner[parent].left == current)
        {
            result += parent * 2;
        }
        else if (inner[parent].right == current)
        {
            result += parent * 2 + 1;
        }
        else
        {
            // Node is neither parent's left nor right child
            // (should never happen)!
            assert(0);
        }

        return result;
    };


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int num_leaves = num_prims;
    int num_inner = num_leaves - 1;

    if (index >= num_leaves)
    {
        return;
    }

    int curr = static_cast<int>(num_inner + index);

    // Insert leaf
    int off_leaf = bvh_node_index(curr, leaves[index].parent);
    bvh_nodes[off_leaf].set_leaf(leaves[index].bbox, prim_refs[index].id, 1);

    if (index >= num_inner)
    {
        return;
    }

    // Assign root
    if (index == 0)
    {
        bvh_nodes[0].set_inner(inner[0].bbox, 1, 0, 0);
        return;
    }

    int off_inner = bvh_node_index(index, inner[index].parent);
//  int off_first = bvh_node_index(inner[index].left, index);
    int off_first = 1 + index * 2; // save some instructions
    bvh_nodes[off_inner].set_inner(inner[index].bbox, off_first, 0, 0);

    auto bbox = bvh_nodes[off_inner].get_bounds();
}

#endif // __CUDACC__

} // lbvh
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
                    static_cast<unsigned>(centroid.x),
                    static_cast<unsigned>(centroid.y),
                    static_cast<unsigned>(centroid.z)
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

    struct split_record
    {
        bool do_split;
        unsigned char axis;
        unsigned char sign;
    };

    // Return true if the leaf should be split into two new leaves. In this case
    // sr.leaves contains the information of the left/right leaves and the
    // method returns true. If the leaf should not be split, returns false.
    template <typename Data>
    split_record split(leaf_infos& childs, leaf_info const& leaf, Data const& /*data*/, int max_leaf_size)
    {
        if (leaf.last - leaf.first <= max_leaf_size)
        {
            return { false, 0, 0 };
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

        return { true, 0, 0 };
    }


#ifdef __CUDACC__

    //-------------------------------------------------------------------------
    // GPU builder based on Karras, Maximizing parallelism in the construction
    // of BVHs octrees and k-d trees (2012).
    //

    thrust::device_vector<detail::lbvh::prim_ref> d_prim_refs;
    thrust::device_vector<aabb> d_prim_bounds;

    template <typename P>
    cuda_index_bvh<P> build(cuda_index_bvh<P> /* */, P* primitives, size_t num_prims)
    {
        using namespace detail::lbvh;

        cuda_index_bvh<P> tree(primitives, num_prims);

        P* first = primitives;
        P* last = primitives + num_prims;


        // Scene and centroid bounding boxes
        aabb invalid;
        invalid.invalidate();
        thrust::device_vector<aabb> bounds(2, invalid);

        aabb* scene_bounds_ptr = thrust::raw_pointer_cast(bounds.data());
        aabb* centroid_bounds_ptr = thrust::raw_pointer_cast(bounds.data() + 1);


        // Compute primitive bounding boxes and centroids
        d_prim_bounds.resize(last - first);
        thrust::device_vector<vec3> centroids(last - first);

        {
            size_t num_threads = 1024;

            compute_bounds_and_centroids<<<div_up(num_prims, num_threads), num_threads>>>(
                    thrust::raw_pointer_cast(d_prim_bounds.data()),
                    thrust::raw_pointer_cast(centroids.data()),
                    scene_bounds_ptr,
                    centroid_bounds_ptr,
                    primitives,
                    num_prims
                    );
        }

        // Compute morton codes for centroids
        d_prim_refs.resize(last - first);

        {
            size_t num_threads = 1024;

            assign_morton_codes<<<div_up(num_prims, num_threads), num_threads>>>(
                    thrust::raw_pointer_cast(d_prim_refs.data()),
                    thrust::raw_pointer_cast(centroids.data()),
                    centroid_bounds_ptr,
                    num_prims
                    );
        }

        // Sort prim refs by morton codes
        thrust::stable_sort(thrust::device, d_prim_refs.begin(), d_prim_refs.end());

        // Use Karras' radix tree algorithm to build hierarchy
        thrust::device_vector<node> inner(num_prims - 1);
        thrust::device_vector<node> leaves(num_prims);

        {
            size_t num_threads = 1024;

            build_hierarchy<<<div_up(num_prims, num_threads), num_threads>>>(
                    thrust::raw_pointer_cast(inner.data()),
                    thrust::raw_pointer_cast(leaves.data()),
                    thrust::raw_pointer_cast(d_prim_refs.data()),
                    num_prims
                    );
        }

        // Expand nodes' bounding boxes by inserting leaves' bounding boxes
        tree.nodes().resize(inner.size() + leaves.size());

        {
            size_t num_threads = 1024;

            assign_node_bounds<<<div_up(num_prims, num_threads), num_threads>>>(
                    thrust::raw_pointer_cast(inner.data()),
                    thrust::raw_pointer_cast(leaves.data()),
                    thrust::raw_pointer_cast(d_prim_bounds.data()),
                    thrust::raw_pointer_cast(d_prim_refs.data()),
                    num_prims
                    );
        }

        // Convert to Visionaray node format (stores the two children
        // of a BVH node next to each other in memory)
        {
            size_t num_threads = 1024;

            collapse<<<div_up(num_prims, num_threads), num_threads>>>(
                    thrust::raw_pointer_cast(tree.nodes().data()),
                    thrust::raw_pointer_cast(inner.data()),
                    thrust::raw_pointer_cast(leaves.data()),
                    thrust::raw_pointer_cast(d_prim_refs.data()),
                    num_prims
                    );
        }

        // Copy primitives to BVH (device to device copy!)
        tree.primitives().resize(num_prims);
        thrust::copy(thrust::device, first, last, tree.primitives().begin());

        // Assign 0,1,2,3,.. indices
        thrust::sequence(thrust::device, tree.indices().begin(), tree.indices().end());

        return tree;
    }

#endif // __CUDACC__


    // TODO:
    bool use_spatial_splits;
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_LBVH_H
