// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_INTERSECTOR_H
#define VSNRAY_INTERSECTOR_H 1

#include <type_traits>
#include <utility>

#include "detail/macros.h"
#include "bvh.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Base type for custom intersectors
//

template <typename Derived>
struct basic_intersector
{
    template <typename R, typename P, typename ...Args>
    VSNRAY_FUNC
    auto operator()(R const& ray, P const& prim, Args&&... args)
        -> decltype( intersect(ray, prim, std::forward<Args>(args)...) )
    {
        return intersect(ray, prim);
    }

    template <typename R, typename P, typename = typename std::enable_if<is_bvh<P>::value>::type>
    VSNRAY_FUNC
    auto operator()(R const& ray, P const& prim)
        -> decltype( intersect(ray, prim, std::declval<Derived&>()) )
    {
        return intersect(ray, prim, *static_cast<Derived*>(this));
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
