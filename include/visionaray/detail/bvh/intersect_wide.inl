// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/matrix.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "../exit_traversal.h"
#include "../stack.h"
#include "../tags.h"
#include "hit_record.h"

namespace visionaray
{

template <
    detail::traversal_type Traversal,
    typename R,
    typename BVH,
    typename Intersector,
    typename T = typename R::scalar_type
    >
VSNRAY_FUNC
inline auto intersect_wide_bvh(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> hit_record_bvh<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >
{
    using namespace detail;
    using HR = hit_record_bvh<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

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

        while (!node.is_leaf())
        {
            hit_record<basic_ray<float>, aabb> hrs[BVH::Width];
            bool bs[BVH::Width];
            int hrs_idx[BVH::Width];
            for (int i = 0; i < node.child_count; ++i)
            {
                hrs[i] = isect(ray, node.get_child_bounds(i), inv_dir);
                bs[i] = is_closer(hrs[i], result, ray.tmin, ray.tmax);
                hrs_idx[i] = i;
            }

            std::sort(hrs_idx, hrs_idx + node.child_count,
                [&](int i, int j) {
                    return (bs[i] && bs[j] && hrs[i].tnear < hrs[j].tnear) ||
                           (bs[i] && !bs[j]);
                });

            if (!bs[hrs_idx[0]])
            {
                goto next;
            }

            for (int i = 1; i < node.child_count; ++i)
            {
                if (!bs[hrs_idx[i]])
                {
                    break;
                }

                st.push(node.children[hrs_idx[i]]);
            }

            node = b.node(node.children[hrs_idx[0]]);
        }

        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        for (auto i = node.get_indices().first; i != node.get_indices().last; ++i)
        {
            auto prim = b.primitive(i);

            auto hr = HR(isect(ray, prim), i);
            auto closer = is_closer(hr, result, ray.tmin, ray.tmax);

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

// overload w/ custom intersector -------------------------

template <typename R, typename BVH, typename Intersector>
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect
        )
    -> decltype(intersect_wide_bvh<detail::ClosestHit>(ray, b, isect))
{
    return intersect_wide_bvh<detail::ClosestHit>(ray, b, isect);
}

// overload w/ default intersector ------------------------

template <typename R, typename BVH>
VSNRAY_FUNC
inline auto intersect_wide_bvh(R const& ray, BVH const& b)
    -> decltype(intersect_wide_bvh<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>())
            )
{
    default_intersector isect;
    return intersect_wide_bvh<detail::ClosestHit>(ray, b, isect);
}

} // visionaray
