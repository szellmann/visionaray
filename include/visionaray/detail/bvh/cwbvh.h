// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_CWBVH_H
#define VSNRAY_DETAIL_BVH_CWBVH_H 1

#include <cassert>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>

#ifdef __CUDACC__
#include "../../cuda/device_vector.h"
#endif

namespace visionaray
{

//--------------------------------------------------------------------------------------------------
// compressed wide BVH node
//

struct cwbvh_node
{
    enum { Width = 8 };

    // grid origin
    vec3 origin;
    // grid scale
    char e[3];
    // imask indicates which of the children
    // are inner nodes
    unsigned char imask;
    unsigned first_child;
    unsigned first_prim;
    unsigned char meta[8];
    unsigned char minx[8];
    unsigned char miny[8];
    unsigned char minz[8];
    unsigned char maxx[8];
    unsigned char maxy[8];
    unsigned char maxz[8];

    VSNRAY_FUNC aabb get_bounds() const
    {
        aabb result;
        result.invalidate();

        for (int i = 0; i < 8; ++i)
        {
            vec3 lower(
                origin.x + (float)minx[i] * powf(2, e[0]),
                origin.y + (float)miny[i] * powf(2, e[1]),
                origin.z + (float)minz[i] * powf(2, e[2])
                );
            result.insert(lower);

            vec3 upper(
                origin.x + (float)maxx[i] * powf(2, e[0]),
                origin.y + (float)maxy[i] * powf(2, e[1]),
                origin.z + (float)maxz[i] * powf(2, e[2])
                );
            result.insert(upper);
        }

        return result;
    }

    VSNRAY_FUNC void set_inner(int index)
    {
        imask |= 1 << index;

        meta[index] = (unsigned char)((1 << 5) | (24 + index));
    }

    VSNRAY_FUNC void set_leaf(int index, int first_prim_relative, int num_prims)
    {
        // max. 3 nodes per child, 24 prims per multi-node
        assert(num_prims >= 1 && num_prims <= 3);
        assert(first_prim_relative >= 0 && first_prim_relative < 24);

        unsigned char unary_encoding = 0b001;
        if (num_prims == 2) unary_encoding = 0b011;
        else if (num_prims == 3) unary_encoding = 0b111;

        meta[index]  = unary_encoding << 5;
        meta[index] |= (unsigned char)first_prim_relative;
    }
};

static_assert(sizeof(cwbvh_node) == 80, "Size mismatch" );

template <
    typename PrimitiveVector,
    typename NodeVector,
    typename = std::enable_if_t<std::is_same_v<typename NodeVector::value_type, cwbvh_node>>
    >
class cwbvh_t
{
public:

    using primitive_type    = typename PrimitiveVector::value_type;
    using primitive_vector  = PrimitiveVector;
    using node_type         = cwbvh_node;
    using node_vector       = NodeVector;

    using bvh_ref  = bvh_ref_t<primitive_type, node_type>;
    using bvh_inst = bvh_inst_t<primitive_type, node_type>;

    enum { Width = 8 };

public:

    primitive_vector const& primitives() const  { return primitives_; }
    primitive_vector&       primitives()        { return primitives_; }

    node_vector const&      nodes() const       { return nodes_; }
    node_vector&            nodes()             { return nodes_; }

    size_t num_primitives() const               { return primitives_.size(); }
    size_t num_nodes() const                    { return nodes_.size(); }

    bvh_ref ref() const
    {
        auto p0 = primitives().data();
        auto p1 = p0 + primitives().size();

        auto n0 = nodes().data();
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

template <typename P>
using cwbvh      = cwbvh_t<aligned_vector<P>, aligned_vector<cwbvh_node, 32>>;

#ifdef __CUDACC__
template <typename P>
using cuda_cwbvh = cwbvh_t<cuda::device_vector<P>, cuda::device_vector<cwbvh_node>>;
#endif


//-------------------------------------------------------------------------------------------------
// Collapser, takes a BVH2, collapses that into a normal BVH8 and then into
// compressed-wide layout:
//

struct cwbvh_collapser : public bvh_collapser
{
    template <typename P>
    void collapse(bvh<P> const& tree, cwbvh<P>& cwbvh_tree, thread_pool& pool)
    {
        // start from uncompressed multi nodes:
        aligned_vector<bvh_multi_node<8>> multi_nodes(tree.num_nodes());

        // create one multi-node for each bvh2 node
        for (size_t i = 0; i < tree.num_nodes(); ++i)
        {
            multi_nodes[i].init(i, tree.nodes().data());
        }

        collapse_multi_nodes(multi_nodes, pool);

        cwbvh_tree.nodes().reserve(multi_nodes.size());

        size_t src_index = 0, dst_index = 0;
        cwbvh_tree.nodes().emplace_back();
        build_cwbvh_layout_rec(multi_nodes, tree.primitives(), cwbvh_tree, src_index, dst_index);
    }

    // recursive function to reorder nodes and prims; both need to be stored in contiguous
    // blocks, so that each cwbvh node can just store the base index to the block, plus a
    // short offset of few bits into that block to locate the child or prim:
    template <typename P>
    void build_cwbvh_layout_rec(
            aligned_vector<bvh_multi_node<8>>& multi_nodes,
            aligned_vector<P> const&           prims,
            cwbvh<P>&                          cwbvh_tree,
            size_t                             src_index,
            size_t                             dst_index
            )
    {
        auto& mn = multi_nodes[src_index];
        cwbvh_node& cn = cwbvh_tree.nodes()[dst_index];

        aabb bbox = mn.get_bounds();

        cn.origin = bbox.min;

        cn.first_child = cwbvh_tree.nodes().size();
        cn.first_prim  = cwbvh_tree.primitives().size();

        cn.imask = 0;

        int children[8];
        for (int i = 0; i < 8; ++i) children[i] = -1;
        int curr_child = 0;

        constexpr int N_q = 8;
        cn.e[0] = (char)(ceilf(log2f((bbox.max.x - cn.origin.x) / (powf(2, N_q) - 1))));
        cn.e[1] = (char)(ceilf(log2f((bbox.max.y - cn.origin.y) / (powf(2, N_q) - 1))));
        cn.e[2] = (char)(ceilf(log2f((bbox.max.z - cn.origin.z) / (powf(2, N_q) - 1))));

        // compute slot assignment:
        float cost[8][8];

        for (int s = 0; s < 8; ++s)
        {
            vec3 dir(
                (s & 0b100) ? -1.0f :1.0f,
                (s & 0b010) ? -1.0f :1.0f,
                (s & 0b001) ? -1.0f :1.0f
                );

            for (int c = 0; c < 8; ++c)
            {
                if (mn.children[c] == INT64_MAX)
                {
                    cost[s][c] = FLT_MAX;
                }
                else
                {
                    cost[s][c] = dot(mn.get_child_bounds(c).center() - bbox.center(), dir);
                }
            }
        }

        int slot_assignment[8];
        bool slot_filled[8];
        for (int i = 0; i < 8; ++i)
        {
            slot_assignment[i] = -1;
            slot_filled[i] = false;
        }

        for (;;)
        {
            float min_cost = FLT_MAX;
            int min_slot = -1;
            int min_index = -1;

            for (int s = 0; s < 8; ++s)
            {
                for (int c = 0; c < 8; ++c)
                {
                    if (slot_assignment[c] == -1)
                    {
                        if (!slot_filled[s] && cost[s][c] < min_cost)
                        {
                            min_cost = cost[s][c];
                            min_slot = s;
                            min_index = c;
                        }
                    }
                }
            }

            if (min_slot == -1)
            {
                break;
            }

            slot_filled[min_slot] = true;
            slot_assignment[min_index] = min_slot;
        }

        for (int c = 0; c < 8; ++c)
        {
            if (slot_assignment[c] == -1)
            {
                for (int s = 0; s < 8; ++s)
                {
                    if (!slot_filled[s])
                    {
                        slot_filled[s] = true;
                        slot_assignment[c] = s;
                        break;
                    }
                }
            }
        }

        // Inverse mapping: child at slot s
        int slot_child[8];
        for (int c = 0; c < 8; ++c)
        {
            slot_child[slot_assignment[c]] = c;
        }

        for (int s = 0; s < 8; ++s)
        {
            int c = slot_child[s];

            // quantize
            cn.minx[s] = floorf((mn.child_bounds.minx[c] - cn.origin.x) / powf(2, cn.e[0]));
            cn.miny[s] = floorf((mn.child_bounds.miny[c] - cn.origin.y) / powf(2, cn.e[1]));
            cn.minz[s] = floorf((mn.child_bounds.minz[c] - cn.origin.z) / powf(2, cn.e[2]));

            cn.maxx[s] = ceilf((mn.child_bounds.maxx[c] - cn.origin.x) / powf(2, cn.e[0]));
            cn.maxy[s] = ceilf((mn.child_bounds.maxy[c] - cn.origin.y) / powf(2, cn.e[1]));
            cn.maxz[s] = ceilf((mn.child_bounds.maxz[c] - cn.origin.z) / powf(2, cn.e[2]));

            cn.meta[s] = 0;  // initially empty

            int64_t addr = mn.children[c];
            if (addr < 0) // leaf!
            {
                // index of first primitive, relative to primitive base index
                // stored by the compressed multi-node:
                int first_prim_relative = cwbvh_tree.primitives().size() - cn.first_prim;
                uint64_t first_prim;
                uint64_t num_prims;
                bvh_multi_node<8>::decode_leaf(addr, first_prim, num_prims);

                if (num_prims > 3)
                {
                    fprintf(
                        stderr,
                        "ERROR: for cwbvh layout prim count *must* be <= 3 but got %llu prims\n",
                        num_prims
                        );
                }

                for (int p = 0; p < num_prims; ++p)
                {
                    cwbvh_tree.primitives().push_back(prims[first_prim + p]);
                }
                cn.set_leaf(s, first_prim_relative, (int)num_prims);
            }
            else if (addr != INT64_MAX)
            {
                int child_id = int(addr);
                children[curr_child++] = child_id;
                cwbvh_tree.nodes().emplace_back();
                cn.set_inner(s);
            }
        }

        // Arrange child nodes as contiguous blocks adjacent in memory:
        for (int i = 0; i < 8; ++i)
        {
            if (children[i] < 0) { assert(i == curr_child); break; }
            size_t dst_index = cn.first_child + i;
            build_cwbvh_layout_rec(multi_nodes, prims, cwbvh_tree, (size_t)children[i], dst_index);
        }
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_CWBVH_H
