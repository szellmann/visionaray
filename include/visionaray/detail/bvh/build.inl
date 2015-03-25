// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>

#include <visionaray/bvh.h>
#include <visionaray/math/aabb.h>

#include "sah.h"


namespace visionaray
{


namespace detail
{


//--------------------------------------------------------------------------------------------------
// insert_indices
//

template <class Tree, class P>
void insert_indices(Tree& tree, P const& pinfos, int first, int last, index_bvh_tag)
{
    auto& indices = tree.indices();

    for (auto i = first; i != last; ++i)
    {
        indices.push_back(pinfos[i].index);
    }
}

template <class Tree, class P>
void insert_indices(Tree& /*tree*/, P const& /*pinfos*/, int /*first*/, int /*last*/, bvh_tag)
{
}


//--------------------------------------------------------------------------------------------------
// permute_primitives
//

template <class Tree, class P, class I>
void permute_primitives(Tree& /*tree*/, P const& /*pinfos*/, I /*first*/, I /*last*/, index_bvh_tag)
{
}

template <class Tree, class P, class I>
void permute_primitives(Tree& tree, P const& pinfos, I first, I last, bvh_tag)
{
    auto& prims = tree.primitives();

    for (auto n = last - first; n--; /**/)
    {
        prims[n] = first[pinfos[n].index];
    }
}


//--------------------------------------------------------------------------------------------------
// build_tree_impl
//

template <class Tree, class P, class Builder>
void build_tree_impl
(
    int         index,
    Tree&       tree,
    P&          pinfos,
    aabb const& prim_bounds,
    aabb const& cent_bounds,
    int         first,
    int         last,
    Builder&    builder,
    int         max_leaf_size
)
{
    // TODO:
    // Make iterative

    auto& nodes = tree.nodes();

    typename Builder::split_result sr;

    auto split = builder.split(sr, pinfos, prim_bounds, cent_bounds, first, last, max_leaf_size);

    if (split)
    {
        auto first_child_index = static_cast<int>(nodes.size());

        nodes[index].set_inner(prim_bounds, first_child_index);

        nodes.emplace_back();
        nodes.emplace_back();

        build_tree_impl
        (
            first_child_index + 0,
            tree,
            pinfos,
            sr.prim_bounds[0],
            sr.cent_bounds[0],
            sr.first,
            sr.middle,
            builder,
            max_leaf_size
        );

        build_tree_impl
        (
            first_child_index + 1,
            tree,
            pinfos,
            sr.prim_bounds[1],
            sr.cent_bounds[1],
            sr.middle,
            sr.last,
            builder,
            max_leaf_size
        );
    }
    else
    {
        nodes[index].set_leaf(prim_bounds, first, last - first);

        insert_indices(tree, pinfos, first, last, typename Tree::tag_type());
    }
}


//--------------------------------------------------------------------------------------------------
// build_tree
//

template <class Tree, class Builder, class I>
void build_tree(Tree& tree, Builder& builder, I first, I last, int max_leaf_size = -1)
{
    if (max_leaf_size <= 0)
    {
        max_leaf_size = 4;
    }

    // Precompute primitive data needed by the builder

    typename Builder::prim_data data(first, last);

    // Preallocate memory
    // Guess number of nodes...

    tree.clear(2 * (data.pinfos.size() / max_leaf_size));

    // Build the tree

    // Create root node
    tree.nodes().emplace_back();

    build_tree_impl
    (
        0, // root node index
        tree,
        data.pinfos,
        data.prim_bounds,
        data.cent_bounds,
        0,
        static_cast<int>(data.pinfos.size()),
        builder,
        max_leaf_size
    );

    // Sort primitives

    permute_primitives(tree, data.pinfos, first, last, typename Tree::tag_type());
}


} // detail


template <typename Tree, typename P>
Tree build(P* primitives, size_t num_prims)
{
    Tree tree(primitives, num_prims);

    detail::binned_sah_builder builder;

    detail::build_tree(tree, builder, primitives, primitives + num_prims);

    return tree;
}


} // visionaray
