// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_GET_NORMAL_H
#define VSNRAY_DETAIL_BVH_GET_NORMAL_H 1

#include <type_traits>
#include <utility>

#include <visionaray/array.h>

// TODO: should not depend on this
#include "hit_record.h"

namespace visionaray
{

namespace detail
{

template <
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename = typename std::enable_if<!simd::is_simd_vector<typename HR::scalar_type>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(HR const& hr, Primitive prim)
    -> decltype(get_normal(static_cast<Base const&>(hr), prim.primitive(hr.primitive_list_index)))
{
    return get_normal(static_cast<Base const&>(hr), prim.primitive(hr.primitive_list_index));
}

template <
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename BaseS = typename decltype( simd::unpack(std::declval<Base const&>()) )::value_type,
    typename V = decltype( get_normal(
            std::declval<BaseS const&>(),
            std::declval<typename Primitive::primitive_type>()
            ) ),
    typename T = typename HR::scalar_type,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(HR const& hr, Primitive prim)
    -> decltype( simd::pack(std::declval<array<V, simd::num_elements<T>::value>>()) )
{
    auto hrs = simd::unpack(hr);

    array<V, simd::num_elements<T>::value> arr;
    for (unsigned i = 0; i < simd::num_elements<T>::value; ++i)
    {
        arr[i] = get_normal(
                static_cast<BaseS const&>(hrs[i]),
                prim.primitive(hrs[i].primitive_list_index)
                );
    }
    return simd::pack(arr);
}

template <
    typename Normals,
    typename HR,
    typename Base = typename HR::base_type,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal_from_bvh(Normals normals, HR const& hr, Primitive /* */)
    -> decltype( get_normal(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{}
            ) )
{
    return get_normal(
            normals,
            static_cast<Base const&>(hr),
            typename Primitive::primitive_type{}
            );
}

} // detail


//-------------------------------------------------------------------------------------------------
// get_normal overloads
//

template <
    typename Normals,
    typename HR,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal(Normals normals, HR const& hr, Primitive prim)
    -> decltype( detail::get_normal_from_bvh(normals, hr, prim) )
{
    return detail::get_normal_from_bvh(normals, hr, prim);
}

template <
    typename HR,
    typename Primitive,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_normal(HR const& hr, Primitive prim)
    -> decltype( detail::get_normal_from_bvh(hr, prim) )
{
    return detail::get_normal_from_bvh(hr, prim);
}


template <
    typename Normals,
    typename HR,
    typename Primitive,
    typename NormalBinding,
    typename = typename std::enable_if<is_any_bvh<Primitive>::value>::type
    >
VSNRAY_FUNC
auto get_shading_normal(
        Normals       normals,
        HR const&     hr,
        Primitive     prim,
        NormalBinding /* */
        )
    -> decltype( detail::get_normal_from_bvh(normals, hr, prim, NormalBinding{}) )
{
    return detail::get_normal_from_bvh(normals, hr, prim, NormalBinding{});
}

} // visionaray

#endif // VSNRAY_DETAIL_BVH_GET_NORMAL_H
