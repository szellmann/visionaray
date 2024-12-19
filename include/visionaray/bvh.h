// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BVH_H
#define VSNRAY_BVH_H 1

#include <cassert>
#include <cstddef>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#ifdef __CUDACC__
#include "cuda/device_vector.h"
#include "cuda/safe_call.h"
#endif

#ifdef __HIPCC__
#include "hip/device_vector.h"
#include "hip/safe_call.h"
#endif

#include "detail/macros.h"
#include "math/aabb.h"
#include "math/forward.h"
#include "math/matrix.h"
#include "aligned_vector.h"
#include "tags.h"

namespace visionaray
{
namespace detail
{

template <typename Container>
inline auto get_pointer(Container const& vec)
    -> decltype(vec.data())
{
    return vec.data();
}

#ifdef __CUDACC__
template <typename T>
inline T const* get_pointer(cuda::device_vector<T> const& vec)
{
    return vec.data();
}
#elif defined(__HIPCC__)
template <typename T>
inline T const* get_pointer(hip::device_vector<T> const& vec)
{
    return vec.data();
}
#endif
} // detail


//--------------------------------------------------------------------------------------------------
// bvh_node
//

struct VSNRAY_ALIGN(32) bvh_node
{
    enum { Width = 2 };

    aabb bbox;
    union
    {
        unsigned first_child;
        unsigned first_prim;
    };
    unsigned short num_prims;
    unsigned char ordered_traversal_axis;
    unsigned char ordered_traversal_sign;

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

    VSNRAY_FUNC unsigned get_first_primitive() const
    {
        assert(is_leaf());
        return first_prim;
    }

    VSNRAY_FUNC unsigned get_num_primitives() const
    {
        assert(is_leaf());
        return static_cast<unsigned>(num_prims);
    }

    VSNRAY_FUNC void set_inner(
            aabb const& bounds, unsigned first_child_index, unsigned char axis, unsigned char sign
            )
    {
        bbox = bounds;
        first_child = first_child_index;
        num_prims = 0;
        ordered_traversal_axis = axis;
        ordered_traversal_sign = sign;
    }

    VSNRAY_FUNC void set_leaf(aabb const& bounds, unsigned first_primitive_index, unsigned count)
    {
        bbox = bounds;
        first_prim = first_primitive_index;
        num_prims = static_cast<unsigned short>(count);
    }
};

static_assert( sizeof(bvh_node) == 32, "Size mismatch" );

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
    return (is_inner(a) && is_inner(b) && a.get_child(0) == b.get_child(0))
        || ( is_leaf(a) && is_leaf(b)  && a.get_first_primitive() == b.get_first_primitive() );
}


//--------------------------------------------------------------------------------------------------
// bvh_multi_node
//

// #define QUANTIZE

template <int W>
struct bvh_multi_node
{
    enum { Width = W };

#ifdef QUANTIZE
    struct {
        unsigned char minx[W];
        unsigned char miny[W];
        unsigned char minz[W];
        unsigned char maxx[W];
        unsigned char maxy[W];
        unsigned char maxz[W];
    } child_bounds;
    vec3 origin;
    vec3 scale;

    template <int I>
    unsigned char quantize(float f) const
    {
        return (f - origin[I]) / scale[I] * 255;
    }

    template <int I>
    float unquantize(unsigned char u) const
    {
        return (u / 255.f) * scale[I] + origin[I];
    }
#else
    struct {
        float minx[W];
        float miny[W];
        float minz[W];
        float maxx[W];
        float maxy[W];
        float maxz[W];
    } child_bounds;
#endif
    int64_t children[Width]; // child[0]: neg(first_prim), child[1]: neg(num_prims)

    void init(unsigned id, bvh_node const* nodes)
    {
        bvh_node const& n = nodes[id];

        for (int i = 0; i < Width; ++i)
        {
            children[i] = INT64_MAX;
#ifdef QUANTIZE
            child_bounds.minx[i] = 255;
            child_bounds.miny[i] = 255;
            child_bounds.minz[i] = 255;
            child_bounds.maxx[i] = 0;
            child_bounds.maxy[i] = 0;
            child_bounds.maxz[i] = 0;
#else
            child_bounds.minx[i] = FLT_MAX;
            child_bounds.miny[i] = FLT_MAX;
            child_bounds.minz[i] = FLT_MAX;
            child_bounds.maxx[i] = -FLT_MAX;
            child_bounds.maxy[i] = -FLT_MAX;
            child_bounds.maxz[i] = -FLT_MAX;
#endif
        }

        if (n.is_inner())
        {
            children[0] = n.first_child;
            children[1] = n.first_child + 1;

#ifdef QUANTIZE
            aabb bounds;
            bounds.invalidate();
            for (int i = 0; i < 2; ++i)
            {
                bounds.insert(nodes[children[i]].get_bounds());
            }

            origin = bounds.min;
            scale = bounds.max - bounds.min;
#endif

            for (int i = 0; i < 2; ++i)
            {
#ifdef QUANTIZE
                child_bounds.minx[i] = quantize<0>(nodes[children[i]].get_bounds().min.x);
                child_bounds.miny[i] = quantize<1>(nodes[children[i]].get_bounds().min.y);
                child_bounds.minz[i] = quantize<2>(nodes[children[i]].get_bounds().min.z);
                child_bounds.maxx[i] = quantize<0>(nodes[children[i]].get_bounds().max.x);
                child_bounds.maxy[i] = quantize<1>(nodes[children[i]].get_bounds().max.y);
                child_bounds.maxz[i] = quantize<2>(nodes[children[i]].get_bounds().max.z);
#else
                child_bounds.minx[i] = nodes[children[i]].get_bounds().min.x;
                child_bounds.miny[i] = nodes[children[i]].get_bounds().min.y;
                child_bounds.minz[i] = nodes[children[i]].get_bounds().min.z;
                child_bounds.maxx[i] = nodes[children[i]].get_bounds().max.x;
                child_bounds.maxy[i] = nodes[children[i]].get_bounds().max.y;
                child_bounds.maxz[i] = nodes[children[i]].get_bounds().max.z;
#endif

                if (nodes[children[i]].is_leaf())
                {
                    bvh_node const& c = nodes[children[i]];
                    uint64_t first_prim = c.get_first_primitive();
                    uint64_t num_prims  = c.get_num_primitives();

                    if (num_prims > 32767)
                    {
                        fprintf(stderr, "ERROR: ignoring leaf with %u prims\n", num_prims);
                        continue;
                    }

                    children[i] = encode_leaf(first_prim, num_prims);
                }
              }
        }
        else if (id == 0 && n.is_leaf())
        {
            // special handling for root node that is a leaf
            // (would be unreachable otherwise)
#ifdef QUANTIZE
            child_bounds.minx[0] = quantize<0>(nodes[0].get_bounds().min.x);
            child_bounds.miny[0] = quantize<1>(nodes[0].get_bounds().min.y);
            child_bounds.minz[0] = quantize<2>(nodes[0].get_bounds().min.z);
            child_bounds.maxx[0] = quantize<0>(nodes[0].get_bounds().max.x);
            child_bounds.maxy[0] = quantize<1>(nodes[0].get_bounds().max.y);
            child_bounds.maxz[0] = quantize<2>(nodes[0].get_bounds().max.z);
#else
            child_bounds.minx[0] = nodes[0].get_bounds().min.x;
            child_bounds.miny[0] = nodes[0].get_bounds().min.y;
            child_bounds.minz[0] = nodes[0].get_bounds().min.z;
            child_bounds.maxx[0] = nodes[0].get_bounds().max.x;
            child_bounds.maxy[0] = nodes[0].get_bounds().max.y;
            child_bounds.maxz[0] = nodes[0].get_bounds().max.z;
#endif
            uint64_t first_prim = n.get_first_primitive();
            uint64_t num_prims  = n.get_num_primitives();
            children[0] = encode_leaf(first_prim, num_prims);
        }
    }

    static uint64_t encode_leaf(uint64_t first_prim, uint64_t num_prims)
    {
        return ~(num_prims << 48 | (first_prim & 0xFFFFFFFFFFFFll));
    }

    static void decode_leaf(uint64_t addr, uint64_t& first_prim, uint64_t& num_prims)
    {
        addr = ~addr;
        first_prim = addr & 0xFFFFFFFFFFFFll;
        num_prims = addr >> 48;
    }

    void collapse_child(bvh_multi_node& child, unsigned dest_id, unsigned source_id)
    {
        children[dest_id] = child.children[source_id];
#ifdef QUANTIZE
        float minx = child.unquantize<0>(child.child_bounds.minx[source_id]);
        float miny = child.unquantize<1>(child.child_bounds.miny[source_id]);
        float minz = child.unquantize<2>(child.child_bounds.minz[source_id]);
        float maxx = child.unquantize<0>(child.child_bounds.maxx[source_id]);
        float maxy = child.unquantize<1>(child.child_bounds.maxy[source_id]);
        float maxz = child.unquantize<2>(child.child_bounds.maxz[source_id]);
#else
        child_bounds.minx[dest_id] = child.child_bounds.minx[source_id];
        child_bounds.miny[dest_id] = child.child_bounds.miny[source_id];
        child_bounds.minz[dest_id] = child.child_bounds.minz[source_id];
        child_bounds.maxx[dest_id] = child.child_bounds.maxx[source_id];
        child_bounds.maxy[dest_id] = child.child_bounds.maxy[source_id];
        child_bounds.maxz[dest_id] = child.child_bounds.maxz[source_id];
#endif
        //child.children[source_id] = INT64_MAX;
    }

    void bounds_as_float(float* dest)
    {
#ifdef QUANTIZE
        for (int i = 0; i < Width; ++i)
        {
            dest[i]             = unquantize<0>(child_bounds.minx[i]);
            dest[i + Width]     = unquantize<1>(child_bounds.miny[i]);
            dest[i + Width * 2] = unquantize<2>(child_bounds.minz[i]);
            dest[i + Width * 3] = unquantize<0>(child_bounds.maxx[i]);
            dest[i + Width * 4] = unquantize<1>(child_bounds.maxy[i]);
            dest[i + Width * 5] = unquantize<2>(child_bounds.maxz[i]);
        }
#else
        memcpy(dest, &child_bounds, sizeof(child_bounds));
#endif
    }

    VSNRAY_FUNC int get_num_children() const
    {
        for (int i = 0; i < Width; ++i)
        {
            if (children[i] == INT64_MAX)
            {
                return i;
            }
        }

        return Width;
    }

    VSNRAY_FUNC aabb get_bounds() const
    {
        aabb result;
        result.invalidate();

        for (int i = 0; i < Width; ++i)
        {
            result.insert(get_child_bounds(i));
        }

        return result;
    }

    VSNRAY_FUNC bool is_inner() const { return children[0] >= 0; }
    VSNRAY_FUNC bool is_leaf() const { return !is_inner(); }

    VSNRAY_FUNC aabb get_child_bounds(unsigned i) const
    {
#ifdef QUANTIZE
        return aabb(
            vec3(
                unquantize<0>(child_bounds.minx[i]),
                unquantize<1>(child_bounds.miny[i]),
                unquantize<2>(child_bounds.minz[i])
                ),
            vec3(
                unquantize<0>(child_bounds.maxx[i]),
                unquantize<1>(child_bounds.maxy[i]),
                unquantize<2>(child_bounds.maxz[i])
                )
            );
#else
        return aabb(
            vec3(child_bounds.minx[i], child_bounds.miny[i], child_bounds.minz[i]),
            vec3(child_bounds.maxx[i], child_bounds.maxy[i], child_bounds.maxz[i])
            );
#endif
    }

    VSNRAY_FUNC unsigned get_child(unsigned i = 0) const
    {
        assert(is_inner());
        assert(i < Width);
        return children[i];
    }

    VSNRAY_FUNC unsigned get_first_primitive() const
    {
        assert(is_leaf());
        return ~children[0];
    }

    VSNRAY_FUNC unsigned get_num_primitives() const
    {
        assert(is_leaf());
        return ~children[1];
    }

    struct index_range
    {
        unsigned first;
        unsigned last;
    };

    VSNRAY_FUNC index_range get_indices() const
    {
        assert(is_leaf());
        return { get_first_primitive(), get_first_primitive() + get_num_primitives() };
    }
};


//--------------------------------------------------------------------------------------------------
// [index_]bvh_ref_t
//

template <typename PrimitiveType, typename Node = bvh_node>
class bvh_ref_t
{
public:

    using primitive_type = PrimitiveType;

    enum { Width = Node::Width };

private:

    using P = const PrimitiveType;
    using N = const Node;

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

    VSNRAY_FUNC P* primitives() const
    {
        return primitives_first;
    }

    VSNRAY_FUNC N* nodes() const
    {
        return nodes_first;
    }

    VSNRAY_FUNC P& primitive(size_t index) const
    {
        return primitives_first[index];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        return nodes_first[index];
    }

    VSNRAY_FUNC bool operator==(bvh_ref_t const& rhs) const
    {
        return primitives_first == rhs.primitives_first
            && primitives_last  == rhs.primitives_last
            && nodes_first      == rhs.nodes_first
            && nodes_last       == rhs.nodes_last;
    }
};

template <typename PrimitiveType, typename Node = bvh_node>
class index_bvh_ref_t
{
public:

    using primitive_type = PrimitiveType;

    enum { Width = Node::Width };

private:

    using P = const PrimitiveType;
    using N = const Node;
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
    VSNRAY_FUNC size_t num_indices() const { return indices_last - indices_first; }

    VSNRAY_FUNC P* primitives() const
    {
        return primitives_first;
    }

    VSNRAY_FUNC N* nodes() const
    {
        return nodes_first;
    }

    VSNRAY_FUNC I* indices() const
    {
        return indices_first;
    }

    VSNRAY_FUNC P& primitive(size_t indirect_index) const
    {
        return primitives_first[indices_first[indirect_index]];
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        return nodes_first[index];
    }

    VSNRAY_FUNC bool operator==(index_bvh_ref_t const& rhs) const
    {
        return primitives_first == rhs.primitives_first
            && primitives_last  == rhs.primitives_last
            && nodes_first      == rhs.nodes_first
            && nodes_first      == rhs.nodes_first
            && indices_last     == rhs.indices_last
            && indices_last     == rhs.indices_last;
    }
};


//--------------------------------------------------------------------------------------------------
// [index_]bvh_inst_t
//

template <typename PrimitiveType, typename Node = bvh_node>
class bvh_inst_t
{
public:

    using primitive_type = PrimitiveType;

    enum { Width = Node::Width };

private:

    using P = const PrimitiveType;
    using N = const Node;

public:

    bvh_inst_t() = default;

    bvh_inst_t(bvh_ref_t<PrimitiveType, Node> const& ref, mat4x3 const& transform)
        : ref_(ref)
        , affine_inv_(inverse(top_left(transform)))
        , trans_inv_(-transform(3))
        , inst_id_(~0u)
    {
    }

    VSNRAY_FUNC size_t num_primitives() const
    {
        return ref_.num_primitives();
    }

    VSNRAY_FUNC size_t num_nodes() const
    {
        return ref_.num_nodes();
    }

    VSNRAY_FUNC P& primitive(size_t index) const
    {
        return ref_.primitive(index);
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        return ref_.node(index);
    }

    VSNRAY_FUNC bvh_ref_t<PrimitiveType, Node> get_ref() const
    {
        return ref_;
    }

    VSNRAY_FUNC mat3 affine_inv() const
    {
        return affine_inv_;
    }

    VSNRAY_FUNC vec3 trans_inv() const
    {
        return trans_inv_;
    }

    VSNRAY_FUNC bool operator==(bvh_inst_t const& rhs) const
    {
        return ref_ == rhs.ref_ && affine_inv_ == rhs.affine_inv_ && trans_inv_ == rhs.trans_inv_;
    }

    template <typename Ray>
    VSNRAY_FUNC void transform_ray(Ray& r) const
    {
        using T = typename Ray::scalar_type;

        matrix<3, 3, T> aff_inv(affine_inv_);
        r.ori = aff_inv * r.ori + vector<3, T>(trans_inv_);
        r.dir = aff_inv * r.dir;
    }

    VSNRAY_FUNC void set_inst_id(int id)
    {
        inst_id_ = id;
    }

    VSNRAY_FUNC int get_inst_id() const
    {
        return inst_id_;
    }

private:

    // BVH ref
    bvh_ref_t<PrimitiveType, Node> ref_;

    // Inverse affine transformation matrix
    mat3 affine_inv_;

    // Inverse translation
    vec3 trans_inv_;

    // Instance ID
    int inst_id_;

};

template <typename PrimitiveType, typename Node = bvh_node>
class index_bvh_inst_t
{
public:

    using primitive_type = PrimitiveType;

    enum { Width = Node::Width };

private:

    using P = const PrimitiveType;
    using N = const Node;

public:

    index_bvh_inst_t() = default;

    index_bvh_inst_t(index_bvh_ref_t<PrimitiveType, Node> const& ref, mat4x3 const& transform)
        : ref_(ref)
        , affine_inv_(inverse(top_left(transform)))
        , trans_inv_(-transform(3))
        , inst_id_(~0u)
    {
    }

    VSNRAY_FUNC size_t num_primitives() const
    {
        return ref_.num_primitives();
    }

    VSNRAY_FUNC size_t num_nodes() const
    {
        return ref_.num_nodes();
    }

    VSNRAY_FUNC size_t num_indices() const
    {
        return ref_.num_indices();
    }

    VSNRAY_FUNC P& primitive(size_t indirect_index) const
    {
        return ref_.primitive(indirect_index);
    }

    VSNRAY_FUNC N& node(size_t index) const
    {
        return ref_.node(index);
    }

    VSNRAY_FUNC index_bvh_ref_t<PrimitiveType, Node> get_ref() const
    {
        return ref_;
    }

    VSNRAY_FUNC mat3 affine_inv() const
    {
        return affine_inv_;
    }

    VSNRAY_FUNC vec3 trans_inv() const
    {
        return trans_inv_;
    }

    VSNRAY_FUNC bool operator==(index_bvh_inst_t const& rhs) const
    {
        return ref_ == rhs.ref_ && affine_inv_ == rhs.affine_inv_ && trans_inv_ == rhs.trans_inv_;
    }

    template <typename Ray>
    VSNRAY_FUNC void transform_ray(Ray& r) const
    {
        using T = typename Ray::scalar_type;

        matrix<3, 3, T> aff_inv(affine_inv_);
        r.ori = aff_inv * (r.ori + vector<3, T>(trans_inv_));
        r.dir = aff_inv * r.dir;
    }

    VSNRAY_FUNC void set_inst_id(int id)
    {
        inst_id_ = id;
    }

    VSNRAY_FUNC int get_inst_id() const
    {
        return inst_id_;
    }

private:

    // BVH ref
    index_bvh_ref_t<PrimitiveType, Node> ref_;

    // Inverse affine transformation matrix
    mat3 affine_inv_;

    // Inverse translation
    vec3 trans_inv_;

    // Instance ID
    int inst_id_;
};


//--------------------------------------------------------------------------------------------------
// [index_]bvh_t
//

template <typename PrimitiveVector, typename NodeVector, int W = 2>
class bvh_t
{
public:

    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_type         = typename NodeVector::value_type;
    using node_vector       = NodeVector;

    using bvh_ref  = bvh_ref_t<primitive_type, node_type>;
    using bvh_inst = bvh_inst_t<primitive_type, node_type>;

    enum { Width = W };

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

    bvh_inst inst(mat4x3 const& transform)
    {
        return bvh_inst(ref(), transform);
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

template <typename PrimitiveVector, typename NodeVector, typename IndexVector, int W = 2>
class index_bvh_t
{
public:

    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_type         = typename NodeVector::value_type;
    using node_vector       = NodeVector;
    using index_vector      = IndexVector;

    using bvh_ref  = index_bvh_ref_t<primitive_type, node_type>;
    using bvh_inst = index_bvh_inst_t<primitive_type, node_type>;

    enum { Width = W };

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
    {
        copy(primitives_, rhs.primitives());
        copy(nodes_, rhs.nodes());
        copy(indices_, rhs.indices());
    }

    primitive_vector const& primitives() const  { return primitives_; }
    primitive_vector&       primitives()        { return primitives_; }

    node_vector const&      nodes() const       { return nodes_; }
    node_vector&            nodes()             { return nodes_; }

    index_vector const&     indices() const     { return indices_; }
    index_vector&           indices()           { return indices_; }

    size_t num_primitives() const               { return primitives_.size(); }
    size_t num_nodes() const                    { return nodes_.size(); }
    size_t num_indices() const                  { return indices_.size(); }

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

    bvh_inst inst(mat4x3 const& transform)
    {
        return bvh_inst(ref(), transform);
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

    template <typename DstVector, typename SrcVector>
    void copy(DstVector& dst, SrcVector const& src)
    {
        dst = src;
    }

#ifdef __CUDACC__
    template <typename T, typename SrcVector>
    void copy(cuda::device_vector<T>& dst, SrcVector const& src)
    {
        dst.resize(src.size());

        CUDA_SAFE_CALL(cudaMemcpy(
            dst.data(),
            src.data(),
            sizeof(T) * src.size(),
            cudaMemcpyHostToDevice
            ));
    }

    template <typename DstVector, typename T>
    void copy(DstVector& dst, cuda::device_vector<T> const& src)
    {
        dst.resize(src.size());

        CUDA_SAFE_CALL(cudaMemcpy(
            dst.data(),
            src.data(),
            sizeof(T) * src.size(),
            cudaMemcpyDeviceToHost
            ));
    }

    template <typename T>
    void copy(cuda::device_vector<T>& dst, cuda::device_vector<T> const& src)
    {
        dst.resize(src.size());

        CUDA_SAFE_CALL(cudaMemcpy(
            dst.data(),
            src.data(),
            sizeof(T) * src.size(),
            cudaMemcpyDeviceToDevice
            ));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
#endif
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
struct is_bvh<bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh : std::false_type {};

template <typename T1, typename T2, typename T3>
struct is_index_bvh<index_bvh_t<T1, T2, T3>> : std::true_type {};

template <typename T>
struct is_index_bvh<index_bvh_ref_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh<index_bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_any_bvh : std::integral_constant<bool, is_bvh<T>::value || is_index_bvh<T>::value> {};


template <typename T>
struct is_bvh_inst : std::false_type {};

template <typename T>
struct is_bvh_inst<bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh_inst : std::false_type {};

template <typename T>
struct is_index_bvh_inst<index_bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_any_bvh_inst : std::integral_constant<bool, is_bvh_inst<T>::value || is_index_bvh_inst<T>::value> {};


//-------------------------------------------------------------------------------------------------
// Typedefs
//

template <typename P>
using bvh               = bvh_t<aligned_vector<P>, aligned_vector<bvh_node, 32>>;
template <typename P>
using index_bvh         = index_bvh_t<aligned_vector<P>, aligned_vector<bvh_node, 32>, aligned_vector<unsigned>>;
template <typename P>
using index_bvh4        = index_bvh_t<aligned_vector<P>, aligned_vector<bvh_multi_node<4>, 32>, aligned_vector<unsigned>, 4>;
template <typename P>
using index_bvh8        = index_bvh_t<aligned_vector<P>, aligned_vector<bvh_multi_node<8>, 32>, aligned_vector<unsigned>, 8>;

#ifdef __CUDACC__
template <typename P>
using cuda_bvh          = bvh_t<cuda::device_vector<P>, cuda::device_vector<bvh_node>>;
template <typename P>
using cuda_index_bvh    = index_bvh_t<cuda::device_vector<P>, cuda::device_vector<bvh_node>, cuda::device_vector<unsigned>>;
#endif

#ifdef __HIPCC__
template <typename P>
using hip_bvh           = bvh_t<hip::device_vector<P>, hip::device_vector<bvh_node>>;
template <typename P>
using hip_index_bvh     = index_bvh_t<hip::device_vector<P>, hip::device_vector<bvh_node>, hip::device_vector<unsigned>>;
#endif


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

#include "detail/bvh/collapse.h"
#include "detail/bvh/get_bounds.inl"
#include "detail/bvh/get_color.h"
#include "detail/bvh/get_normal.h"
#include "detail/bvh/get_tex_coord.h"
#include "detail/bvh/hit_record.h"
#include "detail/bvh/intersect.inl"
#include "detail/bvh/intersect_ray1_bvh4.inl"
#include "detail/bvh/lbvh.h"
#include "detail/bvh/prim_traits.h"
#include "detail/bvh/refit.h"
#include "detail/bvh/sah.h"
#include "detail/bvh/statistics.h"
#include "detail/bvh/traverse.h"

#endif // VSNRAY_BVH_H
