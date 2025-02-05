// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_OPTIMIZE_H
#define VSNRAY_DETAIL_BVH_OPTIMIZE_H 1

#include <vector>

#include "../stack.h"
#include "../thread_pool.h"

//-----------------------------------------------------------------------------
// Based on: http://eastfarthing.com/publications/tree.pdf
// Kensler: Tree Rotations for Improving Bounding Volume Hierarchies
//

namespace visionaray
{

struct bvh_optimizer
{
    template <typename Tree>
    void optimize_tree_rotations(Tree& tree, thread_pool& /*pool*/)
    {
        static_assert(Tree::Width == 2, "Type mismatch");

        #define SA(X) surface_area(X)

        const float C_t = 1.0f; // costs for traversal
        const float C_i = 1.0f; // costs for intersecting a primitive

        // compute SAH costs for each node
        std::vector<float> costs(tree.num_nodes(), FLT_MAX);

        // do this from the bottom up implicitly: only nodes get assigned
        // costs if they're leaves, or if their children have costs assigned.
        // We stop once the root was assigned its costs:
        bool root_costs_assigned = false;
        for (;;)
        {
            for (unsigned i = 0; i < tree.num_nodes(); ++i)
            {
               bvh_node const& n = tree.node(i);
               if (costs[i] < FLT_MAX)
               {
                   continue;
               }

               if (n.is_leaf())
               {
                   costs[i] = C_t + C_i * n.get_num_primitives();
               }
               else
               {
                   unsigned c0 = n.get_child(0);
                   unsigned c1 = n.get_child(1);
                   if (costs[c0] < FLT_MAX && costs[c1] < FLT_MAX)
                   {
                       float sa0 = SA(tree.node(c0).get_bounds());
                       float sa1 = SA(tree.node(c1).get_bounds());
                       costs[i] = C_t + (sa0 * costs[c0] + sa1 * costs[c1]) / SA(n.get_bounds());

                       if (i == 0)
                       {
                           root_costs_assigned = true;
                       }
                   }
               }
            }

            if (root_costs_assigned)
            {
                break;
            }
        }

        detail::stack<64> st;

        unsigned addr = 0;
        st.push(addr);

        int count = 0;
        while (!st.empty())
        {
            auto& node = tree.node(addr);

            // perform rotations:
            if (is_inner(node))
            {
                float C = costs[addr];
                float S = SA(node.get_bounds());

                enum Rotation
                {
                    _00, // exchange c0.0 and c1
                    _01, // exchange c0.1 and c1
                    _11, // exchange c1.1 and c0
                    _10, // exchange c1.0 and c0
                };

                // new costs for the different combinations:
                float c[4] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

                if (is_inner(tree.node(node.get_child(0))))
                {
                    // 0.0 <-> 1
                    {
                        aabb box00 = tree.node(node.get_child(1)).get_bounds();
                        aabb box01 = tree.node(tree.node(node.get_child(0)).get_child(1)).get_bounds();
                        aabb box0 = box00; box0.insert(box01);
                        aabb box1 = tree.node(tree.node(node.get_child(0)).get_child(0)).get_bounds();
                        aabb box = box0; box.insert(box1);

                        float c00 = costs[node.get_child(1)];
                        float c01 = costs[tree.node(node.get_child(0)).get_child(1)];
                        float c0 = C_t + (SA(box00) * c00 + SA(box01) * c01) / SA(box0);
                        float c1 = costs[tree.node(node.get_child(0)).get_child(0)];
                        c[_00] = C_t + (SA(box0) * c0 + SA(box1) * c1) / SA(box);
                    }

                    // 0.1 <-> 1
                    {
                        aabb box00 = tree.node(tree.node(node.get_child(0)).get_child(0)).get_bounds();
                        aabb box01 = tree.node(node.get_child(1)).get_bounds();
                        aabb box0 = box00; box0.insert(box01);
                        aabb box1 = tree.node(tree.node(node.get_child(0)).get_child(1)).get_bounds();
                        aabb box = box0; box.insert(box1);

                        float c00 = costs[tree.node(node.get_child(0)).get_child(0)];
                        float c01 = costs[node.get_child(1)];
                        float c0 = C_t + (SA(box00) * c00 + SA(box01) * c01) / SA(box0);
                        float c1 = costs[tree.node(node.get_child(0)).get_child(1)];
                        c[_01] = C_t + (SA(box0) * c0 + SA(box1) * c1) / SA(box);
                    }
                }

                if (is_inner(tree.node(node.get_child(1))))
                {
                    // 1.1 <-> 0
                    {
                        aabb box10 = tree.node(tree.node(node.get_child(1)).get_child(0)).get_bounds();
                        aabb box11 = tree.node(node.get_child(0)).get_bounds();
                        aabb box1 = box10; box1.insert(box11);
                        aabb box0 = tree.node(tree.node(node.get_child(1)).get_child(1)).get_bounds();
                        aabb box = box0; box.insert(box1);

                        float c10 = costs[tree.node(node.get_child(1)).get_child(0)];
                        float c11 = costs[node.get_child(0)];
                        float c0 = costs[tree.node(node.get_child(1)).get_child(1)];
                        float c1 = C_t + (SA(box10) * c10 + SA(box11) * c11) / SA(box1);
                        c[_11] = C_t + (SA(box0) * c0 + SA(box1) * c1) / SA(box);
                    }

                    // 1.0 <-> 0
                    {
                        aabb box10 = tree.node(node.get_child(0)).get_bounds();
                        aabb box11 = tree.node(tree.node(node.get_child(1)).get_child(1)).get_bounds();
                        aabb box1 = box10; box1.insert(box11);
                        aabb box0 = tree.node(tree.node(node.get_child(1)).get_child(0)).get_bounds();
                        aabb box = box0; box.insert(box1);

                        float c10 = costs[node.get_child(0)];
                        float c11 = costs[tree.node(node.get_child(1)).get_child(1)];
                        float c0 = costs[tree.node(node.get_child(1)).get_child(0)];
                        float c1 = C_t + (SA(box10) * c10 + SA(box11) * c11) / SA(box1);
                        c[_10] = C_t + (SA(box0) * c0 + SA(box1) * c1) / SA(box);
                    }
                }

                int rotation = -1; // -1: don't rotate
                float c_new = C;
                for (int i = 0; i < 4; ++i)
                {
                    if (c[i] < c_new)
                    {
                        rotation = i;
                        c_new = c[i];
                    }
                }

                unsigned n0 = ~0u;
                unsigned n1 = ~0u;

                switch (rotation)
                {
                case _00:
                    n0 = tree.node(node.get_child(0)).get_child(0);
                    n1 = node.get_child(1);
                    break;

                case _01:
                    n0 = tree.node(node.get_child(0)).get_child(1);
                    n1 = node.get_child(1);
                    break;

                case _11:
                    n0 = node.get_child(0);
                    n1 = tree.node(node.get_child(1)).get_child(1);
                    break;

                case _10:
                    n0 = node.get_child(0);
                    n1 = tree.node(node.get_child(1)).get_child(0);
                    break;

                case -1:
                default:
                    break;
                }

                if (n0 != ~0u && n1 != ~0u)
                {
                    bvh_node node0 = tree.node(n0);
                    bvh_node node1 = tree.node(n1);

                    tree.nodes()[n0] = node1;
                    tree.nodes()[n1] = node0;

                    if (rotation == _00 || rotation == _01)
                    {
                        aabb bounds00 = tree.node(tree.node(node.get_child(0)).get_child(0)).get_bounds();
                        aabb bounds01 = tree.node(tree.node(node.get_child(0)).get_child(1)).get_bounds();
                        aabb bounds0 = bounds00; bounds0.insert(bounds01);
                        tree.nodes()[node.get_child(0)].bbox = bounds0;

                        aabb bounds1 = tree.node(node.get_child(1)).get_bounds();
                        aabb bounds = bounds0; bounds.insert(bounds1);
                        tree.nodes()[addr].bbox = bounds;
                    }

                    if (rotation == _11 || rotation == _10)
                    {
                        aabb bounds10 = tree.node(tree.node(node.get_child(1)).get_child(0)).get_bounds();
                        aabb bounds11 = tree.node(tree.node(node.get_child(1)).get_child(1)).get_bounds();
                        aabb bounds1 = bounds10; bounds1.insert(bounds11);
                        tree.nodes()[node.get_child(1)].bbox = bounds1;

                        aabb bounds0 = tree.node(node.get_child(0)).get_bounds();
                        aabb bounds = bounds0; bounds.insert(bounds1);
                        tree.nodes()[addr].bbox = bounds;
                    }

                    count++;
                }
            }

            if (is_inner(node))
            {
                addr = node.get_child(0);
                st.push(node.get_child(1));
            }
            else
            {
                addr = st.pop();
            }
        }

        // printf("rotation count: %i\n", count);

        #undef SA
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_OPTIMIZE_H
