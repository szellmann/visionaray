// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BVH_H
#define VSNRAY_BVH_H

#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#include <cuda.h>
#include <thrust/device_vector.h>
#endif

#include "detail/aligned_vector.h"
#include "math/math.h"

namespace visionaray
{

struct bvh_node
{
    bvh_node() : num_prims(0) {}

    aabb bbox;
    union
    {
        unsigned first_child;
        unsigned first_prim;
    };
    unsigned num_prims;
};


//-------------------------------------------------------------------------------------------------
//
//

VSNRAY_FUNC
inline bool is_inner(bvh_node const& node)
{
    return node.num_prims == 0;
}

VSNRAY_FUNC
inline bool is_leaf(bvh_node const& node)
{
    return node.num_prims != 0;
}


template <typename P>
class bvh
{
public:

    typedef std::size_t size_type;
    typedef aligned_vector<bvh_node> node_vector_type;
    typedef aligned_vector<unsigned> idx_vector_type;

    P const* primitives;
    size_type num_prims;

#ifdef BVH_WITH_GATHER
    explicit bvh(P const* prims, size_t num_prims)
        : primitives(prims)
        , num_prims(num_prims)
        , nodes_(node_vector_type(2 * num_prims - 1))
        , prim_indices_(idx_vector_type(num_prims))
    {
    }
#else
    // TODO!!!!!
    explicit bvh(P const* prims, size_t num_prims)
        : primitives(0)
        , num_prims(num_prims)
        , nodes_(node_vector_type(2 * num_prims - 1))
    {
        VSNRAY_UNUSED(prims);
    }

   ~bvh()
    {
        // FIXME:
        delete[] primitives;
    }
#endif

    bvh_node const* nodes() const { return nodes_.data(); }
    bvh_node* nodes() { return nodes_.data(); }

    bvh_node const* nodes_ptr() const { return nodes_.data(); }
    bvh_node* nodes_ptr() { return nodes_.data(); }

    node_vector_type const& nodes_vector() const { return nodes_; }
    node_vector_type& nodes_vector() { return nodes_; }

    unsigned const* prim_indices() const { return prim_indices_.data(); }
    unsigned* prim_indices() { return prim_indices_.data(); }

    unsigned const* prim_indices_ptr() const { return prim_indices_.data(); }
    unsigned* prim_indices_ptr() { return prim_indices_.data(); }

    idx_vector_type const& prim_indices_vector() const { return prim_indices_; }
    idx_vector_type& prim_indices_vector() { return prim_indices_; }

private:

    node_vector_type nodes_;
    idx_vector_type  prim_indices_;

};


#ifdef __CUDACC__

namespace detail
{

//-------------------------------------------------------------------------------------------------
// Wraps a thrust::device_vector and a raw pointer. The latter is accessible from a CUDA kernel.
// Does only implement the vector interface needed in this scope
//

template <typename T>
class kernel_device_vector
{
public:

    kernel_device_vector() = default;

    template <typename U, typename Alloc>
    /* implicit */ kernel_device_vector(std::vector<U, Alloc> const& host_vec)
        : vector_(host_vec)
        , ptr_(0)
    {
        ptr_ = thrust::raw_pointer_cast(vector_.data());
    }

    VSNRAY_GPU_FUNC T const* data() const { return ptr_; }
    VSNRAY_GPU_FUNC T*       data()       { return ptr_; }

private:

    thrust::device_vector<T> vector_;
    T* ptr_;

};

} // detail


//-------------------------------------------------------------------------------------------------
// BVH for traversal on a CUDA device. Can only be edited on the device.
// Can only be constructed from a host BVH, is not copyable or copy assignable
//

template <typename P>
class device_bvh
{
public:

    P* primitives;

    device_bvh(bvh<P> const& host_bvh)
    {
        cudaError_t err = cudaSuccess;
        err = cudaMalloc( &primitives, host_bvh.num_prims * sizeof(P) );
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("malloc error") + std::to_string(__LINE__));
        }

        err = cudaMemcpy( primitives, host_bvh.primitives, host_bvh.num_prims * sizeof(P),
            cudaMemcpyHostToDevice );
        if (err != cudaSuccess)
        {
            throw std::runtime_error("memcpy error");
        }

        nodes_ = host_bvh.nodes_vector();
        prim_indices_ = host_bvh.prim_indices_vector();
    }

    ~device_bvh()
    {
        cudaError_t err = cudaSuccess;
        err = cudaFree(primitives);
        if (err != cudaSuccess)
        {

        }

    }

    VSNRAY_GPU_FUNC bvh_node const* nodes() const            { return nodes_.data(); }
    VSNRAY_GPU_FUNC bvh_node*       nodes()                  { return nodes_.data(); }

    VSNRAY_GPU_FUNC unsigned const* prim_indices() const     { return prim_indices_.data(); }
    VSNRAY_GPU_FUNC unsigned*       prim_indices()           { return prim_indices_.data(); }

private:

    VSNRAY_NOT_COPYABLE(device_bvh)
    device_bvh& operator=(bvh<P> const& host_bvh);

    detail::kernel_device_vector<bvh_node> nodes_;
    detail::kernel_device_vector<unsigned> prim_indices_;

};

#endif // __CUDACC__


//-------------------------------------------------------------------------------------------------
//
//

template <typename P /* primitive type */>
bvh<P> build(P const* primitives, size_t num_prims);

template <typename T, typename B>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect
(
    basic_ray<T> const& ray,
    B const& b
);

} // visionaray

#include "detail/bvh_build_binned.inl"
#include "detail/bvh_intersect.inl"

#endif


