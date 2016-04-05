// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_NORMAL_H
#define VSNRAY_DETAIL_BVH_GET_NORMAL_H 1

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
auto get_normal(
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim
        )
    -> decltype( get_normal(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            ) )
{
    return get_normal(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            );
}

template <
    typename Normals,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
auto get_normal(
        Normals                             normals,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           /* */,
        NormalBinding                       /* */
        )
    -> decltype( get_normal(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            NormalBinding{} 
            ) )
{
    return get_normal(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            NormalBinding{}
            );
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_NORMAL_H
