// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_TEX_COORD_H
#define VSNRAY_DETAIL_BVH_GET_TEX_COORD_H 1

#include <type_traits>

namespace visionaray
{

template <
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_tex_coord(HR const& hr, Primitive prim)
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
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type,
    typename = typename std::enable_if<!is_any_bvh_inst<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_tex_coord(TexCoords tex_coords, HR const& hr, Primitive prim)
    -> decltype( get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            prim
            ) )
{
    return get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            prim
            );
}

template <
    typename TexCoords,
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type,
    typename = typename std::enable_if<is_any_bvh_inst<Primitive>::value>::type,
    typename = void
    >
VSNRAY_FUNC
auto get_tex_coord(TexCoords tex_coords, HR const& hr, Primitive prim)
    -> decltype( get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            prim
            ) )
{
    return get_tex_coord(
            tex_coords,
            static_cast<Base const&>(hr),
            prim
            );
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_TEX_COORD_H
