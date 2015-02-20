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
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#endif

#include "detail/aligned_vector.h"
#include "math/math.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

template <class T>
T* pointer_cast(T* ptr)
{
    return ptr;
}

#ifdef __CUDACC__
template <class T>
T* pointer_cast(thrust::device_ptr<T> const& ptr)
{
    return thrust::raw_pointer_cast(ptr);
}
#endif
} // detail

//--------------------------------------------------------------------------------------------------
// bvh_node
//

struct bvh_node
{
    bvh_node()
        : num_prims(0)
    {
    }

    aabb bbox;
    union
    {
        unsigned first_child;
        unsigned first_prim;
    };
    unsigned num_prims;
};

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

//--------------------------------------------------------------------------------------------------
// [index_]bvh_ref_t
//

template <typename PrimitiveType>
class bvh_ref_t
{

    using P = const PrimitiveType;
    using N = const bvh_node;

    P* primitives_first;
    P* primitives_last;
    N* nodes_first;
    N* nodes_last;

public:

    bvh_ref_t(P* p0, P* p1, N* n0, N* n1)
        : primitives_first(p0)
        , primitives_last(p1)
        , nodes_first(n0)
        , nodes_last(n1)
    {
    }

    VSNRAY_FUNC size_t num_primitives() const { return primitives_last - primitives_first; }
    VSNRAY_FUNC size_t num_nodes() const { return nodes_last - nodes_first; }

    VSNRAY_FUNC P& primitive(size_t index) const
    {
#if !defined(__CUDA_ARCH__)// || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        assert(index < num_primitives());
#endif

        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
#if !defined(__CUDA_ARCH__)// || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        assert(index < num_nodes());
#endif

        return nodes_first[index];
    }

};

template <typename PrimitiveType>
class index_bvh_ref_t
{

    using P = const PrimitiveType;
    using N = const bvh_node;
    using I = const unsigned;

    P* primitives_first;
    P* primitives_last;
    N* nodes_first;
    N* nodes_last;
    I* indices_first;
    I* indices_last;

public:

    index_bvh_ref_t(P* p0, P* p1, N* n0, N* n1, I* i0, I* i1)
        : primitives_first(p0)
        , primitives_last(p1)
        , nodes_first(n0)
        , nodes_last(n1)
        , indices_first(i0)
        , indices_last(i1)
    {
    }

    VSNRAY_FUNC size_t num_primitives() const { return primitives_last - primitives_first; }
    VSNRAY_FUNC size_t num_nodes() const { return nodes_last - nodes_first; }

    VSNRAY_FUNC P& primitive(size_t indirect_index) const
    {
#if !defined(__CUDA_ARCH__)// || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        assert(indirect_index < indices_last - indices_first);
#endif

        auto index = indices_first[indirect_index];

#if !defined(__CUDA_ARCH__)// || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        assert(index < num_primitives());
#endif

        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
#if !defined(__CUDA_ARCH__)// || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        assert(index < num_nodes());
#endif

        return nodes_first[index];
    }

};

//--------------------------------------------------------------------------------------------------
// [index_]bvh_t
//

template <typename PrimitiveVector, typename NodeVector>
class bvh_t
{
public:

    using tag_type = bvh_tag;

    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_vector       = NodeVector;

    using bvh_ref = bvh_ref_t<primitive_type>;

public:

    bvh_t() = default;

    template <class P>
    explicit bvh_t(P* /*prims*/, size_t count)
        : primitives_(count)
        , nodes_(count == 0 ? 0 : 2 * count - 1)
    {
    }

    template <typename PV, typename NV>
    explicit bvh_t(bvh_t<PV, NV> const& rhs)
        : primitives_(rhs.primitives())
        , nodes_(rhs.nodes())
    {
    }

    primitive_vector const& primitives() const  { return primitives_; }
    primitive_vector&       primitives()        { return primitives_; }

    node_vector const&      nodes() const       { return nodes_; }
    node_vector&            nodes()             { return nodes_; }

    bvh_ref ref() const
    {
        auto p0 = detail::pointer_cast(primitives().data());
        auto p1 = p0 + primitives().size();

        auto n0 = detail::pointer_cast(nodes().data());
        auto n1 = n0 + nodes().size();

        return { p0, p1, n0, n1 };
    }

private:

    primitive_vector primitives_;
    node_vector nodes_;

};

template <typename PrimitiveVector, typename NodeVector, typename IndexVector>
class index_bvh_t
{
public:

    using tag_type = index_bvh_tag;

    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_vector       = NodeVector;
    using index_vector      = IndexVector;

    using bvh_ref = index_bvh_ref_t<primitive_type>;

public:

    index_bvh_t() = default;

    template <class P>
    explicit index_bvh_t(P* prims, size_t count)
        : primitives_(prims, prims + count)
        , nodes_(count == 0 ? 0 : 2 * count - 1)
        , indices_(count)
    {
    }

    template <typename PV, typename NV, typename IV>
    explicit index_bvh_t(index_bvh_t<PV, NV, IV> const& rhs)
        : primitives_(rhs.primitives())
        , nodes_(rhs.nodes())
        , indices_(rhs.indices())
    {
    }

    primitive_vector const& primitives() const  { return primitives_; }
    primitive_vector&       primitives()        { return primitives_; }

    node_vector const&      nodes() const       { return nodes_; }
    node_vector&            nodes()             { return nodes_; }

    index_vector const&     indices() const     { return indices_; }
    index_vector&           indices()           { return indices_; }

    bvh_ref ref() const
    {
        auto p0 = detail::pointer_cast(primitives().data());
        auto p1 = p0 + primitives().size();

        auto n0 = detail::pointer_cast(nodes().data());
        auto n1 = n0 + nodes().size();

        auto i0 = detail::pointer_cast(indices().data());
        auto i1 = i0 + indices().size();

        return { p0, p1, n0, n1, i0, i1 };
    }

private:

    primitive_vector primitives_;
    node_vector nodes_;
    index_vector indices_;

};

template <typename P> using bvh                 = bvh_t<aligned_vector<P>, aligned_vector<bvh_node>>;
template <typename P> using index_bvh           = index_bvh_t<aligned_vector<P>, aligned_vector<bvh_node>, aligned_vector<unsigned>>;

#ifdef __CUDACC__
template <typename P> using device_bvh          = bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>>;
template <typename P> using device_index_bvh    = index_bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>, thrust::device_vector<unsigned>>;
#endif

//-------------------------------------------------------------------------------------------------
//
//

template <typename BvhType, typename P>
BvhType build(P* primitives, size_t num_prims);

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


