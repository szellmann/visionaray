// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>

#include <visionaray/bvh.h>
#include <visionaray/math/aabb.h>

#include "sah.h"
#include "../algorithm.h"


namespace visionaray
{


namespace detail
{


//--------------------------------------------------------------------------------------------------
// build_tree_impl
//

template <typename Nodes, typename Indices, typename Builder, typename LeafInfo, typename Data>
void build_tree_impl(
        int             index,
        Nodes&          nodes,
        Indices&        indices,
        Builder&        builder,
        LeafInfo const& leaf,
        Data const&     data,
        int             max_leaf_size
        )
{
    typename Builder::leaf_infos childs;

    auto split = builder.split(childs, leaf, data, max_leaf_size);

    if (split)
    {
        auto first_child_index = static_cast<int>(nodes.size());

        nodes[index].set_inner(leaf.prim_bounds, first_child_index);

        nodes.emplace_back();
        nodes.emplace_back();

        // Construct right subtree
        build_tree_impl(
                first_child_index + 1,
                nodes,
                indices,
                builder,
                childs[1],
                data,
                max_leaf_size
                );

        // Construct left subtree
        build_tree_impl(
                first_child_index + 0,
                nodes,
                indices,
                builder,
                childs[0],
                data,
                max_leaf_size
                );
    }
    else
    {
        auto first = static_cast<int>(indices.size());
        auto count = builder.insert_indices(indices, leaf);

        nodes[index].set_leaf(leaf.prim_bounds, first, count);
    }
}


//--------------------------------------------------------------------------------------------------
// build_tree
//

template <typename Tree, typename Builder, typename Root, typename I>
void build_tree_work(Tree& tree, Builder& builder, Root root, I first, I /*last*/, int max_leaf_size, std::true_type/*is_index_bvh*/)
{
    build_tree_impl(
            0, // root node index
            tree.nodes(),
            tree.indices(),
            builder,
            root,
            first, // primitive data
            max_leaf_size
            );
}

template <typename Tree, typename Builder, typename Root, typename I>
void build_tree_work(Tree& tree, Builder& builder, Root root, I first, I /*last*/, int max_leaf_size, std::false_type/*is_index_bvh*/)
{
    // TODO:
    // Maybe rewrite the builder to directly shuffle the primitives?!?!

    std::vector<unsigned> indices;

    //assert(builder.use_spatial_splits == false);

    auto uss = builder.use_spatial_splits;

    builder.use_spatial_splits = false;

    build_tree_impl(
            0, // root node index
            tree.nodes(),
            indices,
            builder,
            root,
            first, // primitive data
            max_leaf_size
            );

    builder.use_spatial_splits = uss;

    assert(indices.size() == tree.primitives().size());

    // Reorder the primitives according to the indices.
    algo::reorder_n(indices.begin(), tree.primitives().begin(), indices.size());
}

template <typename Tree, typename Builder, typename I>
void build_tree(Tree& tree, Builder& builder, I first, I last, int max_leaf_size = -1)
{
    if (max_leaf_size <= 0)
    {
        max_leaf_size = 4;
    }

    // Precompute primitive data needed by the builder

    auto root = builder.init(first, last);

    // Preallocate memory
    // Guess number of nodes...

    auto count = std::distance(first, last);

    tree.clear(2 * (count / max_leaf_size));

    // Build the tree

    // Create root node
    tree.nodes().emplace_back();

    build_tree_work(tree, builder, root, first, last, max_leaf_size, is_index_bvh<Tree>());
}


} // detail


template <typename Tree, typename P>
Tree build(P* primitives, size_t num_prims, bool enable_spatial_splits)
{
    Tree tree(primitives, num_prims);

    detail::sah_builder builder;

    builder.enable_spatial_splits(enable_spatial_splits);
    builder.set_alpha(1.0e-5f);

    detail::build_tree(tree, builder, primitives, primitives + num_prims);

    return tree;
}


} // visionaray
