// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>
#include <utility>

#include <visionaray/math/math.h>
#include <visionaray/update_if.h>

#include "../exit_traversal.h"
#include "../stack.h"
#include "../tags.h"
#include "hit_record.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Ray / BVH intersection
//

template <
    detail::traversal_type Traversal,
    typename T,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename Intersector,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        basic_ray<T> const& ray,
        BVH const&          b,
        Intersector&        isect,
        T                   max_t = numeric_limits<T>::max(),
        Cond                update_cond = Cond()
        )
    -> hit_record_bvh<
        basic_ray<T>,
        BVH,
        decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
        >
{

    using namespace detail;
    using HR = hit_record_bvh<
        basic_ray<T>,
        BVH,
        decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
        >;

    HR result;

    stack<32> st;
    st.push(0); // address of root node

    auto inv_dir = T(1.0) / ray.dir;

    // while ray not terminated
next:
    while (!st.empty())
    {
        auto node = b.node(st.pop());

        // while node does not contain primitives
        //     traverse to the next node

        while (!is_leaf(node))
        {
            auto children = &b.node(node.first_child);

            auto hr1 = isect(ray, children[0].bbox, inv_dir);
            auto hr2 = isect(ray, children[1].bbox, inv_dir);

            auto b1 = any( update_cond(hr1, result, max_t) );
            auto b2 = any( update_cond(hr2, result, max_t) );

            if (b1 && b2)
            {
                unsigned near_addr = all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                st.push(node.first_child + (!near_addr));
                node = b.node(node.first_child + near_addr);
            }
            else if (b1)
            {
                node = b.node(node.first_child);
            }
            else if (b2)
            {
                node = b.node(node.first_child + 1);
            }
            else
            {
                goto next;
            }
        }


        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        auto begin = node.first_prim;
        auto end   = node.first_prim + node.num_prims;

        for (auto i = begin; i != end; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = update_cond(hr, result, max_t);

#ifndef __CUDA_ARCH__
            if (!any(closer))
            {
                continue;
            }
#endif

            update_if(result, hr, closer);

            exit_traversal<Traversal> early_exit;
            if (early_exit.check(result))
            {
                return result;
            }
        }
    }

    return result;

}


//-------------------------------------------------------------------------------------------------
// Default intersect returns closest hit!
//

template <
    typename T,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename Intersector,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        basic_ray<T> const& ray,
        BVH const&          b,
        Intersector&        isect,
        Cond                update_cond = Cond()
        )
    -> hit_record_bvh<
        basic_ray<T>,
        BVH,
        decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
        >
{
    return intersect<detail::ClosestHit>(ray, b, isect, update_cond);
}

} // visionaray
