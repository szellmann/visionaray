// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include "../stack.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Traverse whole acceleration data structures
//
//
// Pass the acceleration data structure and a parameter func(B::node_type) to traversal functions
//
// Traversal functions:
//  - traverse_depth_first()
//  - traverse_leaves()
//
// TODO:
//  - traverse_breadth_first()
//
//
//-------------------------------------------------------------------------------------------------



template <typename B, typename F>
void traverse_depth_first(B const& b, F func)
{
    detail::stack<64> st;

    unsigned addr = 0;
    st.push(addr);

    while (!st.empty())
    {
        auto node = b.node(addr);

        func(node);

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
}

template <typename B, typename F>
void traverse_leaves(B const& b, F func)
{
    traverse_depth_first(b, [&](typename B::node_type const& node)
    {
        if (is_leaf(node))
        {
            func(node);
        }
    });
}


//-------------------------------------------------------------------------------------------------
// Traverse acceleration structure starting from nodes
//
//
// Pass the acceleration data structure, the node to start traversal from, and a
// parameter func(B::node_type) to traversal functions
//
// Traversal functions:
//  - traverse_parents()
//
// TODO:
//  - traverse_breadth_first()
//  - traverse_depth_first()
//  - traverse_leaves()
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Traverse parent nodes
// Complexity O(n), n = node count
//

template <typename B, typename N, typename F>
void traverse_parents(B const& b, N const& node, F func)
{
    detail::stack<128> parents;

    auto nit = std::find(b.nodes().rbegin(), b.nodes().rend(), node);

    if (nit != b.nodes().rend())
    {
        unsigned addr = b.nodes().rend() - nit - 1;

        for (auto it = nit; it != b.nodes().crend(); ++it)
        {
            if (is_inner(*it) && (addr == it->get_child(0) || addr == it->get_child(1)))
            {
                func(*it);
                addr = b.nodes().rend() - it - 1;
            }
        }
    }
}

} // visionaray
