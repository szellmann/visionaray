// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_COLLAPSE_H
#define VSNRAY_DETAIL_BVH_COLLAPSE_H 1

#include "../stack.h"
#include "../thread_pool.h"

//-----------------------------------------------------------------------------
// Wide tree collapser implementation from (Sec. 4.2):
// https://graphics.stanford.edu/~boulos/papers/multi_rt08.pdf
//-----------------------------------------------------------------------------
namespace visionaray
{

struct bvh_collapser
{
    template <typename Tree, typename WideTree>
    void collapse(Tree const& tree, WideTree& wide_tree, thread_pool& /*pool*/)
    {
        static_assert(Tree::Width == 2, "Type mismatch");

        auto& multi_nodes = wide_tree.nodes();
        multi_nodes.resize(tree.num_nodes());

        // create one multi-node for each bvh2 node
        for (size_t i = 0; i < tree.num_nodes(); ++i)
        {
            bvh_node n = tree.node(i);
            multi_nodes[i].init(i, detail::get_pointer(tree.nodes()));
        }

#if 1
        detail::stack<64> st;

        st.push(0);

        while (!st.empty())
        {
            unsigned addr = st.pop();
            auto& node = multi_nodes[addr];

            while (node.get_num_children() < WideTree::Width)
            {
                int best_child_id = -1;
                float best_sa = 0.0f;

                for (int i = 0; i < node.get_num_children(); ++i)
                {
                    if (node.children[i] < 0)
                    {
                        continue;
                    }

                    const auto& child = multi_nodes[node.children[i]];

                    int inner_nodes = 0;
                    for (int c = 0; c < child.get_num_children(); ++c)
                    {
                        if (child.children[c] > 0)
                        {
                            inner_nodes++;
                        }
                    }

                    // Don't collapse leaves into root; multi-nodes
                    // cannot be leaves!
                    if (inner_nodes == 0 && addr == 0)
                    {
                        continue;
                    }

                    // Child bounds are stored inside this node!
                    const aabb& child_bounds = node.get_child_bounds(i);

                    // Check if we can accommodate all grand children:
                    if (node.get_num_children() - 1 + child.get_num_children() <= WideTree::Width)
                    {
                        float sa = surface_area(child_bounds);
                        if (sa > best_sa)
                        {
                            best_child_id = i;
                            best_sa = sa;
                        }
                    }
                }

                // no valid child: stop searching
                if (best_child_id == -1)
                {
                    break;
                }

                // Collapse:
                auto& best_child = multi_nodes[node.children[best_child_id]];

                // move best child's first child up into its new slot:
                node.collapse_child(best_child, best_child_id, 0);
                // Append the remaining children to the end of the list (if any):
                unsigned child_id = node.get_num_children();
                for (int i = 1; i < best_child.get_num_children(); ++i)
                {
                    node.collapse_child(best_child, child_id++, i);
                }
            }

            // Recurse:
            for (int i = 0; i < node.get_num_children(); ++i)
            {
                if (node.children[i] > 0)
                {
                    st.push(node.children[i]);
                }
            }
        }
#endif

        // Remove empty nodes
        std::vector<int64_t> prefix(multi_nodes.size());
        prefix[0] = 0;

        for (size_t i = 1; i < prefix.size(); ++i)
        {
            int has_children = multi_nodes[i].is_empty() ? 0 : 1;
            prefix[i] = prefix[i - 1] + has_children;
        }

        multi_nodes.erase(std::remove_if(
                multi_nodes.begin(),
                multi_nodes.end(),
                [](const auto& node) { return node.is_empty(); }
                ),
            multi_nodes.end()
            );

        for (size_t i = 0; i < multi_nodes.size(); ++i)
        {
            auto& node = multi_nodes[i];

            for (int c = 0; c < node.get_num_children(); ++c)
            {
                if (node.children[c] >= 0)
                {
                    node.children[c] = prefix[node.children[c]];
                }
            }
        }

        // Assign rest of the tree
        init_primitives(tree, wide_tree);
    }

private:
    template <typename Tree, typename P, typename N, int W>
    void init_primitives(Tree const& tree, bvh_t<P, N, W>& wide_tree)
    {
        wide_tree.primitives() = tree.primitives();
    }

    template <typename Tree, typename P, typename N, typename U, int W>
    void init_primitives(Tree const& tree, index_bvh_t<P, N, U, W>& wide_tree)
    {
        wide_tree.primitives() = tree.primitives();
        wide_tree.indices() = tree.indices();
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_COLLAPSE_H
