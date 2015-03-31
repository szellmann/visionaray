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
// build_tree_impl
//

template <typename Tree, typename Builder, typename LeafInfo, typename Data>
void build_tree_impl(
        int             index,
        Tree&           tree,
        Builder&        builder,
        LeafInfo const& leaf,
        Data const&     data,
        int             max_leaf_size
        )
{
    auto& nodes = tree.nodes();
    auto& indices = tree.indices();

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
                tree,
                builder,
                childs[1],
                data,
                max_leaf_size
                );

        // Construct left subtree
        build_tree_impl(
                first_child_index + 0,
                tree,
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

template <typename Tree, typename Builder, typename I>
void build_tree(Tree& tree, Builder& builder, I first, I last, int max_leaf_size = -1 )
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

    build_tree_impl(
            0, // root node index
            tree,
            builder,
            root,
            first, // primitive data
            max_leaf_size
            );
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
