// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_COLOR_H
#define VSNRAY_DETAIL_BVH_GET_COLOR_H 1

#include <type_traits>

#include <visionaray/bvh.h>

#include "hit_record.h"

namespace visionaray
{

template <
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim
        )
    -> decltype( get_color(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            ) )
{
    return get_color(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            );
}

template <
    typename Colors,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename ColorBinding,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        Colors                              colors,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           /* */,
        ColorBinding                        /* */
        )
    -> decltype( get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            ColorBinding{} 
            ) )
{
    return get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            ColorBinding{}
            );
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_COLOR_H
