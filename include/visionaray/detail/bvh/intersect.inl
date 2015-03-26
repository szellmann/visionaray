// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../stack.h"

namespace visionaray
{

template <typename T, typename B>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect
(
    basic_ray<T> const& ray,
    B const& b
)
{

    using namespace detail;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.hit = T(0.0);
    result.t = T(numeric_limits<float>::max());
    result.prim_id = T(0.0);

    stack<32> st;
    st.push(0); // address of root node

    auto inv_dir = T(1.0) / ray.dir;

    // while ray not terminated
next:
    while (!st.empty())
    {
        auto addr = st.pop();

        // while node does not contain primitives
        //     traverse to the next node

#ifdef __CUDA_ARCH__
        bool search_leaf = true;
#endif
        for (;;)
        {
            auto const& node = b.node(addr);

            if (is_leaf(node))
            {
#ifdef __CUDA_ARCH__
                search_leaf = false;
#else
                break;
#endif
            }
            else
            {
            auto children = &b.node(node.first_child);

            auto hr1 = intersect(ray, children[0].bbox, inv_dir);
            auto hr2 = intersect(ray, children[1].bbox, inv_dir);

            auto b1 = any( hr1.hit && hr1.tnear < result.t && hr1.tfar >= T(0.0) );
            auto b2 = any( hr2.hit && hr2.tnear < result.t && hr2.tfar >= T(0.0) );

            if (b1 && b2)
            {
                unsigned near_addr = all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                st.push(node.first_child + (!near_addr));
                addr = node.first_child + near_addr;
            }
            else if (b1)
            {
                addr = node.first_child;
            }
            else if (b2)
            {
                addr = node.first_child + 1;
            }
            else
            {
                goto next;
            }
            }

#ifdef __CUDA_ARCH__
            if (!__any(search_leaf))
            {
                break;
            }
#endif
        }


        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        auto const& node = b.node(addr);

        auto begin = node.first_prim;
        auto end   = node.first_prim + node.num_prims;

        for (auto i = begin; i != end; ++i)
        {
            auto prim = b.primitive(i);

            auto hr  = intersect(ray, prim);
            auto closer = hr.hit && ( hr.t >= T(0.0) && hr.t < result.t );

#ifndef __CUDA_ARCH__
            if (!any(closer))
            {
                continue;
            }
#endif

            result.hit |= closer;
            result.t = select( closer, hr.t, result.t );
            result.prim_type = select( closer, hr.prim_type, result.prim_type );
            result.prim_id   = select( closer, hr.prim_id, result.prim_id );
            result.geom_id   = select( closer, hr.geom_id, result.geom_id );
            result.u         = select( closer, hr.u, result.u );
            result.v         = select( closer, hr.v, result.v );
        }
    }

    return result;

}

} // visionaray
