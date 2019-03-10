// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/limits.h>
#include <visionaray/math/matrix.h>
#include <visionaray/intersector.h>
#include <visionaray/update_if.h>

#include "../exit_traversal.h"
#include "../multi_hit.h"
#include "../stack.h"
#include "../tags.h"
#include "../traversal_result.h"
#include "hit_record.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Ray / BVH intersection
//

template <
    detail::traversal_type Traversal,
    size_t MultiHitMax = 1,             // Max hits for multi-hit traversal
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<!is_any_bvh_inst<BVH>::value>::type,
    typename Intersector,
    typename T = typename R::scalar_type,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect,
        T            max_t = numeric_limits<T>::max(),
        Cond         update_cond = Cond()
        )
    -> typename detail::traversal_result< hit_record_bvh<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >, Traversal, MultiHitMax>::type
{

    using namespace detail;
    using HR = hit_record_bvh<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    using RT = typename detail::traversal_result<HR, Traversal, MultiHitMax>::type;

    RT result;

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
            auto children = &b.node(node.get_child(0));

            auto hr1 = isect(ray, children[0].get_bounds(), inv_dir);
            auto hr2 = isect(ray, children[1].get_bounds(), inv_dir);

            auto b1 = any( is_closer(hr1, result, max_t) );
            auto b2 = any( is_closer(hr2, result, max_t) );

            if (b1 && b2)
            {
                unsigned near_addr = all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                st.push(node.get_child(!near_addr));
                node = b.node(node.get_child(near_addr));
            }
            else if (b1)
            {
                node = b.node(node.get_child(0));
            }
            else if (b2)
            {
                node = b.node(node.get_child(1));
            }
            else
            {
                goto next;
            }
        }


        // while node contains untested primitives
        //     perform a ray-primitive intersection test

        for (auto i = node.get_indices().first; i != node.get_indices().last; ++i)
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


// Overload for instances ---------------------------------

template <
    detail::traversal_type Traversal,
    size_t MultiHitMax = 1,             // Max hits for multi-hit traversal
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh_inst<BVH>::value>::type,
    typename Intersector,
    typename T = typename R::scalar_type,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect,
        T            max_t = numeric_limits<T>::max(),
        Cond         update_cond = Cond()
        )
    -> typename detail::traversal_result< hit_record_bvh_inst<
            R,
            decltype( isect(ray, std::declval<typename BVH::primitive_type>()) )
            >, Traversal, MultiHitMax>::type
{
    using namespace detail;
    using HR = hit_record_bvh_inst<R, decltype(isect(ray, std::declval<typename BVH::primitive_type>()))>;

    using RT = typename detail::traversal_result<HR, Traversal, MultiHitMax>::type;

    R transformed_ray = ray;
    transformed_ray.ori = (matrix<4, 4, T>(b.transform_inv()) * vector<4, T>(ray.ori, T(1.0))).xyz();
    transformed_ray.dir = (matrix<4, 4, T>(b.transform_inv()) * vector<4, T>(ray.dir, T(0.0))).xyz();
    // NOTE: dir is in general *not* normalized!

    auto hr = intersect<Traversal, MultiHitMax>(
            transformed_ray,
            b.get_ref(),
            isect,
            max_t,
            update_cond
            );

    return RT(hr, hr.primitive_list_index, matrix<4, 4, T>(b.transform_inv()));
}


//-------------------------------------------------------------------------------------------------
// Default intersect returns closest hit!
//

// overload w/ custom intersector -------------------------

template <
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename Intersector,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        R const&     ray,
        BVH const&   b,
        Intersector& isect,
        Cond         update_cond = Cond()
        )
    -> decltype(intersect<detail::ClosestHit>(ray, b, isect, numeric_limits<typename R::scalar_type>::max(), update_cond))
{
    return intersect<detail::ClosestHit>(ray, b, isect, numeric_limits<typename R::scalar_type>::max(), update_cond);
}

// overload w/ default intersector ------------------------

template <
    typename R,
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename Cond = is_closer_t
    >
VSNRAY_FUNC
inline auto intersect(
        R const&   ray,
        BVH const& b,
        Cond       update_cond = Cond()
        )
    -> decltype(intersect<detail::ClosestHit>(
            ray,
            b,
            std::declval<default_intersector&>(), numeric_limits<typename R::scalar_type>::max(), update_cond)
            )
{
    default_intersector isect;
    return intersect<detail::ClosestHit>(ray, b, isect, numeric_limits<typename R::scalar_type>::max(), update_cond);
}


} // visionaray
