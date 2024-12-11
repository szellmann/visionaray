// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_INTERSECTOR_H
#define VSNRAY_INTERSECTOR_H 1

#include <cstddef>
#include <type_traits>
#include <utility>

#include "detail/macros.h"
#include "detail/tags.h"
#include "bvh.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Base type for custom intersectors
//

template <typename Derived>
struct basic_intersector
{
    template <size_t N>
    using multi_hit_max = std::integral_constant<size_t, N>;


    template <typename R, typename P, typename ...Args>
    VSNRAY_FUNC
    auto operator()(R const& ray, P const& prim, Args&&... args)
        -> decltype( intersect(ray, prim, std::forward<Args>(args)...) )
    {
        return intersect(ray, prim, std::forward<Args>(args)...);
    }


    // BVH ------------------------------------------------

    template <typename R, typename P, typename = typename std::enable_if<is_any_bvh<P>::value>::type>
    VSNRAY_FUNC
    auto operator()(R const& ray, P const& prim)
        -> decltype( intersect(ray, prim, std::declval<Derived&>()) )
    {
        return intersect(ray, prim, *static_cast<Derived*>(this));
    }


    // BVH any hit ----------------------------------------

    template <
        typename R,
        typename P,
        typename = typename std::enable_if<is_any_bvh<P>::value>::type
        >
    VSNRAY_FUNC
    auto operator()(
            detail::any_hit_tag     /* */,
            multi_hit_max<1>        /* */,
            R const&                ray,
            P const&                prim
            )
        -> decltype( intersect<detail::AnyHit>(ray, prim, std::declval<Derived&>()) )
    {
        return intersect<detail::AnyHit>(ray, prim, *static_cast<Derived*>(this));
    }


    // BVH closest hit ------------------------------------

    template <
        typename R,
        typename P,
        typename = typename std::enable_if<is_any_bvh<P>::value>::type
        >
    VSNRAY_FUNC
    auto operator()(
            detail::closest_hit_tag /* */,
            multi_hit_max<1>        /* */,
            R const&                ray,
            P const&                prim
            )
        -> decltype( intersect<detail::ClosestHit>(ray, prim, std::declval<Derived&>()) )
    {
        return intersect<detail::ClosestHit>(ray, prim, *static_cast<Derived*>(this));
    }


    // BVH multi hit --------------------------------------

    template <
        size_t   N,
        typename R,
        typename P,
        typename = typename std::enable_if<is_any_bvh<P>::value>::type
        >
    VSNRAY_FUNC
    auto operator()(
            detail::multi_hit_tag   /* */,
            multi_hit_max<N>        /* */,
            R const&                ray,
            P const&                prim
            )
        -> decltype( intersect<detail::MultiHit, N>(ray, prim, std::declval<Derived&>()) )
    {
        return intersect<detail::MultiHit, N>(ray, prim, *static_cast<Derived*>(this));
    }
};


//-------------------------------------------------------------------------------------------------
// Default intersector
//

struct default_intersector : basic_intersector<default_intersector>
{
};

} // visionaray

#endif // VSNRAY_INTERSECTOR_H
