// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_CUDA_BVH_LBVH_H
#define VSNRAY_DETAIL_CUDA_BVH_LBVH_H 1

#include <cub/cub.cuh>

#include <visionaray/cuda/device_vector.h>
#include <visionaray/cuda/safe_call.h>

#include "lbvh-common.h"
#include "lbvh-kernels.h"

namespace visionaray
{
namespace cuda
{

struct lbvh_builder
{
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


    //-------------------------------------------------------------------------
    // GPU builder based on Karras, Maximizing parallelism in the construction
    // of BVHs octrees and k-d trees (2012).
    //

    cuda::device_vector<detail::lbvh::prim_ref> d_prim_refs;
    cuda::device_vector<aabb> d_prim_bounds;

    cudaStream_t copy_stream{0};

    lbvh_builder()
    {
        CUDA_SAFE_CALL(cudaStreamCreate(&copy_stream));
    }

    ~lbvh_builder()
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(copy_stream));
    }

    template <typename BVH, typename P>
    BVH build(BVH /* */, P* primitives, size_t num_prims)
    {
        using namespace detail::lbvh;

        BVH tree(primitives, num_prims);

        if (primitives == nullptr || num_prims == 0)
        {
            return tree;
        }

        P* first = primitives;
        P* last = primitives + num_prims;


        // Scene and centroid bounding boxes
        aabb invalid;
        invalid.invalidate();
        cuda::device_vector<aabb> bounds(2, invalid);

        aabb* scene_bounds_ptr = bounds.data();
        aabb* centroid_bounds_ptr = bounds.data() + 1;


        // Compute primitive bounding boxes and centroids
        d_prim_bounds.resize(last - first);
        cuda::device_vector<vec3> centroids(last - first);

        {
            size_t num_threads = 1024;

            compute_bounds_and_centroids<<<div_up(num_prims, num_threads), num_threads>>>(
                    d_prim_bounds.data(),
                    centroids.data(),
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
                    d_prim_refs.data(),
                    centroids.data(),
                    centroid_bounds_ptr,
                    num_prims
                    );
        }

        // Sort prim refs by morton codes
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceMergeSort::StableSortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_prim_refs.data(),
            d_prim_refs.size(),
            detail::CustomLess()
            );
        CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceMergeSort::StableSortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_prim_refs.data(),
            d_prim_refs.size(),
            detail::CustomLess()
            );
        CUDA_SAFE_CALL(cudaFree(d_temp_storage));

        // Use Karras' radix tree algorithm to build hierarchy
        cuda::device_vector<node> inner(num_prims - 1);
        cuda::device_vector<node> leaves(num_prims);

        {
            size_t num_threads = 1024;

            size_t num_inner = inner.size();
            if (num_inner > 0)
            {
                init_nodes<<<div_up(num_inner, num_threads), num_threads>>>(
                        inner.data(),
                        num_inner
                        );
            }

            size_t num_leaves = leaves.size();
            assert(num_leaves > 0); // should have at least one leaf node!
            init_nodes<<<div_up(num_leaves, num_threads), num_threads>>>(
                    leaves.data(),
                    num_leaves
                    );
        }

        {
            size_t num_threads = 1024;

            build_hierarchy<<<div_up(num_prims, num_threads), num_threads>>>(
                    inner.data(),
                    leaves.data(),
                    d_prim_refs.data(),
                    num_prims
                    );
        }

        // Expand nodes' bounding boxes by inserting leaves' bounding boxes
        tree.nodes().resize(inner.size() + leaves.size());

        {
            size_t num_threads = 1024;

            assign_node_bounds<<<div_up(num_prims, num_threads), num_threads>>>(
                    inner.data(),
                    leaves.data(),
                    d_prim_bounds.data(),
                    d_prim_refs.data(),
                    num_prims
                    );
        }

        // Convert to Visionaray node format (stores the two children
        // of a BVH node next to each other in memory)
        {
            size_t num_threads = 1024;

            collapse<<<div_up(num_prims, num_threads), num_threads>>>(
                    tree.nodes().data(),
                    inner.data(),
                    leaves.data(),
                    d_prim_refs.data(),
                    num_prims
                    );
        }

        // Copy primitives to BVH (device to device copy!)
        tree.primitives().resize(num_prims);
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            tree.primitives().begin(),
            first,
            num_prims * sizeof(P),
            cudaMemcpyDefault,
            copy_stream
            ));
        CUDA_SAFE_CALL(cudaStreamSynchronize(copy_stream));

        assign_indices(tree);

        return tree;
    }

    template <typename P>
    void assign_indices(cuda_index_bvh<P>& tree)
    {
        using namespace detail::lbvh;

        // assign 0,1,2,3,.. indices
        {

            size_t num_threads = 1024;

            sequence<<<div_up(tree.indices().size(), num_threads), num_threads>>>(
                    tree.indices().data(),
                    tree.indices().size()
                    );
        }
    }

    template <typename P>
    void assign_indices(cuda_bvh<P>&)
    {
      // no-op
    }




    // TODO:
    bool use_spatial_splits;
};

} // cuda
} // visionaray

#endif // VSNRAY_DETAIL_CUDA_BVH_LBVH_H
