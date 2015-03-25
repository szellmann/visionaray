// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../stack.h"

namespace visionaray
{

template <typename B, typename F>
void traverse_depth_first(B const& b, F func)
{
    detail::stack<64> st;
    static unsigned const Sentinel = unsigned(-1);
    st.push(Sentinel);

    unsigned addr = 0;
    auto node = b.node(addr);

    for (;;)
    {
        func(node);

        if (is_inner(node))
        {
            addr = node.first_child;
            st.push(node.first_child + 1);
        }
        else
        {
            addr = st.pop();
        }

        if (addr != Sentinel)
        {
            node = b.node(addr);
        }
        else
        {
            break;
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

} // visionaray
