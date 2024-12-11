// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_COLLAPSE_H
#define VSNRAY_DETAIL_BVH_COLLAPSE_H 1

#include <iostream>
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

        unsigned addr = 0;
        st.push(addr);

        while (!st.empty())
        {
            addr = st.pop();
            auto node = multi_nodes[addr];

            while (node.child_count < WideTree::Width)
            {
                int best_child_id = -1;
                float best_sa = 0.0f;

                for (int i = 0; i < node.child_count; ++i)
                {
                    const auto& child = multi_nodes[node.children[i]];

                    if (!child.is_valid() || child.is_leaf())
                    {
                        continue;
                    }

                    // Child bounds are stored inside this node!
                    const aabb& child_bounds = node.get_child_bounds(i);

                    // Check if we can accommodate all grand children:
                    if (node.child_count -1 + child.child_count <= WideTree::Width)
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
                node.children[best_child_id] = best_child.children[0];
                node.bbox[best_child_id] = best_child.get_child_bounds(0);
                // Append the remaining children to the end of the list (if any):
                for (int i = 1; i < best_child.child_count; ++i)
                {
                    node.children[node.child_count] = best_child.children[i];
                    node.bbox[node.child_count] = best_child.get_child_bounds(i);
                    node.child_count++;
                }
            }

            // Recurse:
            for (int i = 0; i < node.child_count; ++i)
            {
                unsigned child_id = node.children[i];
                const auto& child = multi_nodes[child_id];
                if (!child.is_valid() || child.is_leaf())
                {
                    continue;
                }
                st.push(child_id);
            }
        }

        // multi_nodes.erase(std::remove_if(
        //         multi_nodes.begin(),
        //         multi_nodes.end(),
        //         [](const auto& node) { return !node.is_valid(); }
        //         ),
        //     multi_nodes.end()
        //     );
#endif
        
        wide_tree.primitives() = tree.primitives();
        wide_tree.indices() = tree.indices();
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_COLLAPSE_H
