// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_COLOR_H
#define VSNRAY_DETAIL_BVH_GET_COLOR_H 1

#include <type_traits>

#include <visionaray/tags.h>

#include "hit_record.h"

namespace visionaray
{

template <
    typename R,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        hit_record_bvh<R, Base> const& hr,
        Primitive                      prim
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

//
// TODO:
// The overloads below are essentially equivalent but for the color binding parameter
// When being less specific about the color binding, overload resolution will however
// consider the non-compound primitives' get_color() functions as candidates because
// those are overloaded based on color binding
//

// colors per face ----------------------------------------

template <
    typename Colors,
    typename R,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        Colors                         colors,
        hit_record_bvh<R, Base> const& hr,
        Primitive                           /* */,
        colors_per_face_binding             /* */
        )
    -> decltype( get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_face_binding{}
            ) )
{
    return get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_face_binding{}
            );
}

// colors per vertex --------------------------------------

template <
    typename Colors,
    typename R,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        Colors                         colors,
        hit_record_bvh<R, Base> const& hr,
        Primitive                           /* */,
        colors_per_vertex_binding           /* */
        )
    -> decltype( get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_vertex_binding{}
            ) )
{
    return get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_vertex_binding{}
            );
}

// colors per geometry ------------------------------------

template <
    typename Colors,
    typename R,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_color(
        Colors                         colors,
        hit_record_bvh<R, Base> const& hr,
        Primitive                           /* */,
        colors_per_geometry_binding         /* */
        )
    -> decltype( get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_geometry_binding{}
            ) )
{
    return get_color(
            colors,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            colors_per_geometry_binding{}
            );
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_COLOR_H
