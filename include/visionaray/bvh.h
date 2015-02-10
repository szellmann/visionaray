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
}

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
// [indexed_]bvh_ref_t
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
        assert(index < num_primitives());

        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        assert(index < num_nodes());

        return nodes_first[index];
    }
};

template <typename PrimitiveType>
class indexed_bvh_ref_t
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
    indexed_bvh_ref_t(P* p0, P* p1, N* n0, N* n1, I* i0, I* i1)
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
        assert(indirect_index < indices_last - indices_first);

        auto index = indices_first[indirect_index];

        assert(index < num_primitives());

        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        assert(index < num_nodes());

        return nodes_first[index];
    }
};

//--------------------------------------------------------------------------------------------------
// [indexed_]bvh_t
//

template <typename PrimitiveVector, typename NodeVector>
class bvh_t
{
public:
    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_vector       = NodeVector;

    using bvh_ref = bvh_ref_t<primitive_type>;

public:
    bvh_t() = default;

    explicit bvh_t(primitive_type const* /*prims*/, size_t count)
        : primitives_(count)
        , nodes_(count == 0 ? 0 : 2*count - 1)
    {
        assert(count != 0);
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

        return {p0, p1, n0, n1};
    }

private:
    primitive_vector primitives_;
    node_vector nodes_;
};

template <typename PrimitiveVector, typename NodeVector, typename IndexVector>
class indexed_bvh_t
{
public:
    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_vector       = NodeVector;
    using index_vector      = IndexVector;

    using bvh_ref = indexed_bvh_ref_t<primitive_type>;

public:
    indexed_bvh_t() = default;

    explicit indexed_bvh_t(primitive_type const* /*prims*/, size_t count)
        : primitives_(count)
        , nodes_(count == 0 ? 0 : 2*count - 1)
        , indices_(count)
    {
        assert(count != 0);
    }

    template <typename PV, typename NV, typename IV>
    explicit indexed_bvh_t(indexed_bvh_t<PV, NV, IV> const& rhs)
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

        return {p0, p1, n0, n1, i0, i1};
    }

private:
    primitive_vector primitives_;
    node_vector nodes_;
    index_vector indices_;
};

template <typename P> using bvh                 = bvh_t<aligned_vector<P>, aligned_vector<bvh_node>>;
template <typename P> using indexed_bvh         = indexed_bvh_t<aligned_vector<P>, aligned_vector<bvh_node>, aligned_vector<unsigned>>;

#ifdef __CUDACC__
template <typename P> using device_bvh          = bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>>;
template <typename P> using indexed_device_bvh  = indexed_bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>, thrust::device_vector<unsigned>>;
#endif

//-------------------------------------------------------------------------------------------------
//
//

template <typename P /* primitive type */>
bvh<P> build(P const* primitives, size_t num_prims, bvh_tag);

template <typename P>
indexed_bvh<P> build(P const* primitives, size_t num_prims, indexed_bvh_tag);

template <typename P>
auto build(P const* primitives, size_t num_prims)
    -> decltype( build(primitives, num_prims, bvh_tag()) )
{
    return build(primitives, num_prims, bvh_tag());
}

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


