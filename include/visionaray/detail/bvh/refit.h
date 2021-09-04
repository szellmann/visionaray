// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_REFIT_H
#define VSNRAY_DETAIL_BVH_REFIT_H 1

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

#include <visionaray/aligned_vector.h>

#include "../parallel_for.h"
#include "../range.h"
#include "../stack.h"
#include "../thread_pool.h"

namespace visionaray
{

struct bvh_refitter
{
    template <typename Tree, typename P>
    void refit(Tree& tree, P* primitives, size_t num_prims, thread_pool& pool)
    {
        static_assert(is_index_bvh<Tree>::value, "Type mismatch");

        std::copy(primitives, primitives + num_prims, tree.primitives().data());

        // Generate primitive bounds
        aligned_vector<aabb> prim_bounds(num_prims);

        parallel_for(
            pool,
            tiled_range1d<size_t>(0, num_prims, 16),
            [&](range1d<size_t> const& r)
            {
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    prim_bounds[i] = get_bounds(primitives[i]);
                }
            });

        // Reassign node bounding boxes
        std::mutex mtx;

        parallel_for(
            pool,
            tiled_range1d<size_t>(0, tree.num_nodes(), 16),
            [&](range1d<size_t> const& r)
            {
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    detail::stack<64> st;

                    aabb bbox;
                    bbox.invalidate();

                    unsigned addr = static_cast<unsigned>(i);

                    st.push(addr);

                    while (!st.empty())
                    {
                        auto node = tree.node(addr);

                        if (node.is_inner())
                        {
                            addr = node.get_child(0);
                            st.push(node.get_child(1));
                        }
                        else
                        {
                            auto indices = node.get_indices();

                            for (unsigned j = indices.first; j != indices.last; ++j)
                            {
                                bbox.insert(prim_bounds[tree.indices()[j]]);
                            }

                            addr = st.pop();
                        }
                    }

                    bvh_node n = tree.node(i);

                    if (n.is_inner())
                    {
                        tree.nodes()[i].set_inner(
                                bbox,
                                n.get_child(0),
                                n.ordered_traversal_axis,
                                n.ordered_traversal_sign
                                );
                    }
                    else
                    {
                        tree.nodes()[i].set_leaf(bbox, n.get_first_primitive(), n.get_num_primitives());
                    }
                }
            });
    }

    template <typename Tree, typename P>
    void refit(Tree& tree, P* primitives, size_t num_prims)
    {
        // TODO: this freezes when the pool is not static
        // and the function is called repeatedly
        static thread_pool pool(std::thread::hardware_concurrency());

        refit(tree, primitives, num_prims, pool);
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_REFIT_H
