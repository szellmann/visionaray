// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_TEX_COORD_H
#define VSNRAY_DETAIL_BVH_GET_TEX_COORD_H 1

#include <type_traits>

#include <visionaray/bvh.h>

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
auto get_tex_coord(
        hit_record_bvh<R, Base> const& hr,
        Primitive                      prim
        )
    -> decltype( get_tex_coord(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            ) )
{
    return get_tex_coord(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            );
}

template <
    typename TexCoords,
    typename R,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_tex_coord(
        TexCoords                      tex_coords,
        hit_record_bvh<R, Base> const& hr,
        Primitive                      /* */
        )
    -> decltype( get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{}
            ) )
{
    return get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{}
            );
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_TEX_COORD_H
