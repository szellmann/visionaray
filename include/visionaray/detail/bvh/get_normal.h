// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_NORMAL_H
#define VSNRAY_DETAIL_BVH_GET_NORMAL_H 1

#include <type_traits>
#include <utility>

#include <visionaray/array.h>
#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>
#include <visionaray/prim_traits.h>

#include "hit_record.h"

namespace visionaray
{

namespace detail
{

template <
    typename NormalFunc,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<!simd::is_simd_vector<
        typename hit_record_bvh<R, BVH, Base>::scalar_type>::value>::type,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim
        )
    -> decltype( std::declval<NormalFunc>()(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            ) )
{
    NormalFunc t;
    return t(
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index)
            );
}

template <
    typename NormalFunc,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename BaseS = typename decltype( simd::unpack(std::declval<Base const&>()) )::value_type,
    typename V = decltype( std::declval<NormalFunc>()(
            std::declval<BaseS const&>(),
            std::declval<typename Primitive::primitive_type>()
            ) ),
    typename T = typename hit_record_bvh<R, BVH, Base>::scalar_type,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_CPU_FUNC
auto get_normal_from_bvh(
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim
        )
    -> decltype( simd::pack(std::declval<array<V, simd::num_elements<T>::value>>()) )
{
    NormalFunc func;

    auto hrs = simd::unpack(hr);

    array<V, simd::num_elements<T>::value> arr;
    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        arr[i] = func(
                static_cast<BaseS const&>(hrs[i]),
                prim.primitive(hrs[i].primitive_list_index)
                );
    }
    return simd::pack(arr);
}

template <
    typename NormalFunc,
    typename Normals,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(
        Normals                             normals,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           /* */,
        NormalBinding                       /* */,
        typename std::enable_if<num_normals<typename Primitive::primitive_type, NormalBinding>::value == 1>::type* = 0
        )
    -> decltype( std::declval<NormalFunc>()(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            NormalBinding{}
            ) )
{
    NormalFunc t;
    return t(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            NormalBinding{}
            );
}

template <
    typename NormalFunc,
    typename Normals,
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(
        Normals                             normals,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim,
        NormalBinding                       /* */,
        typename std::enable_if<num_normals<typename Primitive::primitive_type, NormalBinding>::value >= 2>::type* = 0
        )
    -> decltype( std::declval<NormalFunc>()(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{},
            NormalBinding{}
            ) )
{
    NormalFunc t;
    return t(
            normals,
            static_cast<Base const&>(hr),
            prim.primitive(hr.primitive_list_index),
            NormalBinding{}
            );
}

} // detail


//-------------------------------------------------------------------------------------------------
// get_normal overloads
//

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
VSNRAY_FUNC
auto get_normal(
        Normals                             normals,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim,
        NormalBinding                       /* */
        )
    -> decltype( detail::get_normal_from_bvh<detail::get_normal_t>(normals, hr, prim, NormalBinding{}) )
{
    return detail::get_normal_from_bvh<detail::get_normal_t>(normals, hr, prim, NormalBinding{});
}

template <
    typename R,
    typename BVH,
    typename Base,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal(
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim
        )
    -> decltype( detail::get_normal_from_bvh<detail::get_normal_t>(hr, prim) )
{
    return detail::get_normal_from_bvh<detail::get_normal_t>(hr, prim);
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
VSNRAY_FUNC
auto get_shading_normal(
        Normals                             normals,
        hit_record_bvh<R, BVH, Base> const& hr,
        Primitive                           prim,
        NormalBinding                       /* */
        )
    -> decltype( detail::get_normal_from_bvh<detail::get_shading_normal_t>(normals, hr, prim, NormalBinding{}) )
{
    return detail::get_normal_from_bvh<detail::get_shading_normal_t>(normals, hr, prim, NormalBinding{});
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_NORMAL_H
