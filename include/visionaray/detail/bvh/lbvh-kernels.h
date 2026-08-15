// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_LBVH_KERNELS_H
#define VSNRAY_DETAIL_BVH_LBVH_KERNELS_H 1

#include "lbvh-common.h"

namespace visionaray
{
namespace detail
{

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


//-------------------------------------------------------------------------------------------------
// Kernels
//

namespace lbvh
{

//-------------------------------------------------------------------------------------------------
// Node data structure only used for construction w/ Karras' algorithm. Alignment is bad, but
// node has parent pointer!
//

struct node
{
    VSNRAY_GPU_FUNC void init()
    {
        bbox = aabb(vec3(numeric_limits<float>::max()), vec3(-numeric_limits<float>::max()));
        left = -1;
        right = -1;
        parent = -1;
    }

    aabb bbox;
    int left;
    int right;
    int parent;
};


//-------------------------------------------------------------------------------------------------
// GPU kernels (CUDA and HIP)
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
                static_cast<unsigned>(centroid.x),
                static_cast<unsigned>(centroid.y),
                static_cast<unsigned>(centroid.z)
                );
    }
}

static __global__ void init_nodes(node* nodes, size_t num_nodes)
{
    size_t index = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

    if (index < num_nodes)
    {
        nodes[index].init();
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

    while (inner && next >= 0)
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
    if (leaves[index].parent >= 0 && num_inner > 0)
    {
        int off_leaf = bvh_node_index(curr, leaves[index].parent);
        bvh_nodes[off_leaf].set_leaf(leaves[index].bbox, prim_refs[index].id, 1);
    }
    else
    {
        // Leaf itself is the root node!
        bvh_nodes[0].set_leaf(leaves[index].bbox, prim_refs[0].id, 1);
        return;
    }

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

static __global__ void sequence(
        unsigned* indices,    // OUT: 0,1,2,.. indices
        unsigned  num_indices // IN:  number of indices
        )
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_indices)
    {
        return;
    }

    indices[index] = index;
}

} // lbvh
} // detail
} // visionaray

#endif // VSNRAY_DETAIL_BVH_LBVH_KERNELS_H
