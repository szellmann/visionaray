// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BVH_H
#define VSNRAY_BVH_H 1

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __CUDACC__
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#endif

#include "math/aabb.h"
#include "aligned_vector.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

template <typename Container>
auto get_pointer(Container const& vec)
    -> decltype(vec.data())
{
    return vec.data();
}

#ifdef __CUDACC__
template <typename T>
T const* get_pointer(thrust::device_vector<T> const& vec)
{
    return thrust::raw_pointer_cast(vec.data());
}
#endif
} // detail


//--------------------------------------------------------------------------------------------------
// bvh_node
//

struct bvh_node
{
    aabb bbox;
    union
    {
        unsigned first_child;
        unsigned first_prim;
    };
    unsigned num_prims;

    VSNRAY_FUNC bool is_inner() const { return num_prims == 0; }
    VSNRAY_FUNC bool is_leaf() const { return num_prims != 0; }

    VSNRAY_FUNC aabb const& get_bounds() const
    {
        return bbox;
    }

    VSNRAY_FUNC unsigned get_child(unsigned i = 0) const
    {
        assert(is_inner());
        return first_child + i;
    }

    struct index_range
    {
        unsigned first;
        unsigned last;
    };

    VSNRAY_FUNC index_range get_indices() const
    {
        assert(is_leaf());
        return { first_prim, first_prim + num_prims };
    }

    VSNRAY_FUNC unsigned get_num_primitives() const
    {
        assert(is_leaf());
        return num_prims;
    }

    VSNRAY_FUNC void set_inner(aabb const& bounds, unsigned first_child_index)
    {
        bbox = bounds;
        first_child = first_child_index;
        num_prims = 0;
    }

    VSNRAY_FUNC void set_leaf(aabb const& bounds, unsigned first_primitive_index, unsigned count)
    {
        assert(count > 0);

        bbox = bounds;
        first_prim = first_primitive_index;
        num_prims = count;
    }
};

VSNRAY_FUNC
inline bool is_inner(bvh_node const& node)
{
    return node.is_inner();
}

VSNRAY_FUNC
inline bool is_leaf(bvh_node const& node)
{
    return node.is_leaf();
}

VSNRAY_FUNC
inline bool operator==(bvh_node const& a, bvh_node const& b)
{
    return (is_inner(a) && is_inner(b) && a.first_child == b.first_child)
        || ( is_leaf(a) && is_leaf(b)  &&  a.first_prim == b.first_prim );
}


//--------------------------------------------------------------------------------------------------
// [index_]bvh_ref_t
//

template <typename PrimitiveType>
class bvh_ref_t
{
public:

    using primitive_type = PrimitiveType;

private:

    using P = const PrimitiveType;
    using N = const bvh_node;

    P* primitives_first;
    P* primitives_last;
    N* nodes_first;
    N* nodes_last;

public:

    bvh_ref_t() = default;

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
        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        return nodes_first[index];
    }

};

template <typename PrimitiveType>
class index_bvh_ref_t
{
public:

    using primitive_type = PrimitiveType;

private:

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

    index_bvh_ref_t() = default;

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
        return primitives_first[indices_first[indirect_index]];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
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
    using node_type         = typename NodeVector::value_type;
    using node_vector       = NodeVector;

    using bvh_ref = bvh_ref_t<primitive_type>;

public:

    bvh_t() = default;

    template <typename P>
    explicit bvh_t(P* prims, size_t count)
        : primitives_(prims, prims + count)
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

    size_t num_primitives() const               { return primitives_.size(); }
    size_t num_nodes() const                    { return nodes_.size(); }

    bvh_ref ref() const
    {
        auto p0 = detail::get_pointer(primitives());
        auto p1 = p0 + primitives().size();

        auto n0 = detail::get_pointer(nodes());
        auto n1 = n0 + nodes().size();

        return { p0, p1, n0, n1 };
    }

    primitive_type const& primitive(size_t index) const
    {
        return primitives_[index];
    }

    node_type const& node(size_t index) const
    {
        return nodes_[index];
    }

    void clear(size_t capacity = 0)
    {
        nodes_.clear();
        nodes_.reserve(capacity);
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
    using node_type         = typename NodeVector::value_type;
    using node_vector       = NodeVector;
    using index_vector      = IndexVector;

    using bvh_ref = index_bvh_ref_t<primitive_type>;

public:

    index_bvh_t() = default;

    template <typename P>
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

    size_t num_primitives() const               { return primitives_.size(); }
    size_t num_nodes() const                    { return nodes_.size(); }

    bvh_ref ref() const
    {
        auto p0 = detail::get_pointer(primitives());
        auto p1 = p0 + primitives().size();

        auto n0 = detail::get_pointer(nodes());
        auto n1 = n0 + nodes().size();

        auto i0 = detail::get_pointer(indices());
        auto i1 = i0 + indices().size();

        return { p0, p1, n0, n1, i0, i1 };
    }

    primitive_type const& primitive(size_t indirect_index) const
    {
        return primitives_[indices_[indirect_index]];
    }

    node_type const& node(size_t index) const
    {
        return nodes_[index];
    }

    void clear(size_t capacity = 0)
    {
        nodes_.clear();
        nodes_.reserve(capacity);

        indices_.clear();
        indices_.reserve(capacity);
    }

private:

    primitive_vector primitives_;
    node_vector nodes_;
    index_vector indices_;

};


//-------------------------------------------------------------------------------------------------
// bvh traits
//

template <typename T>
struct is_bvh : std::false_type {};

template <typename T1, typename T2>
struct is_bvh<bvh_t<T1, T2>> : std::true_type {};

template <typename T>
struct is_bvh<bvh_ref_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh : std::false_type {};

template <typename T1, typename T2, typename T3>
struct is_index_bvh<index_bvh_t<T1, T2, T3>> : std::true_type {};

template <typename T>
struct is_index_bvh<index_bvh_ref_t<T>> : std::true_type {};

template <typename T>
struct is_any_bvh : std::integral_constant<bool, is_bvh<T>::value || is_index_bvh<T>::value>
{
};


//-------------------------------------------------------------------------------------------------
// Typedefs
//

template <typename P>
using bvh               = bvh_t<aligned_vector<P>, aligned_vector<bvh_node>>;
template <typename P>
using index_bvh         = index_bvh_t<aligned_vector<P>, aligned_vector<bvh_node>, aligned_vector<unsigned>>;

#ifdef __CUDACC__
template <typename P>
using cuda_bvh          = bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>>;
template <typename P>
using cuda_index_bvh    = index_bvh_t<thrust::device_vector<P>, thrust::device_vector<bvh_node>, thrust::device_vector<unsigned>>;
#endif


//-------------------------------------------------------------------------------------------------
// build() interface
//

template <typename Tree, typename P>
Tree build(P* primitives, size_t num_prims, bool use_spatial_splits = false);


//-------------------------------------------------------------------------------------------------
// Traversal algorithms
//
// NOTE: use intersect(ray, bvh) for *ray* / bvh traversal
//

template <typename B, typename F>
void traverse_depth_first(B const& b, F func);

template <typename B, typename F>
void traverse_leaves(B const& b, F func);

template <typename B, typename N, typename F>
void traverse_parents(B const& b, N const& n, F func);

} // visionaray

#include "detail/bvh/build.inl"
#include "detail/bvh/get_bounds.inl"
#include "detail/bvh/get_color.h"
#include "detail/bvh/get_normal.h"
#include "detail/bvh/get_tex_coord.h"
#include "detail/bvh/intersect.inl"
#include "detail/bvh/prim_traits.h"
#include "detail/bvh/traverse.inl"

#endif // VSNRAY_BVH_H
