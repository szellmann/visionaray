// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BVH_TRAITS_H
#define VSNRAY_BVH_TRAITS_H 1

#include <type_traits>

#include "bvh.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// bvh traits
//

template <typename T>
struct is_bvh : std::false_type {};

template <typename T1, typename T2>
struct is_bvh<bvh_t<T1, T2>> : std::true_type {};

template <typename T>
struct is_bvh<bvh_ref_t<T>> : std::true_type {};

template <typename T>
struct is_bvh<bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh : std::false_type {};

template <typename T1, typename T2, typename T3>
struct is_index_bvh<index_bvh_t<T1, T2, T3>> : std::true_type {};

template <typename T>
struct is_index_bvh<index_bvh_ref_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh<index_bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_any_bvh : std::integral_constant<bool, is_bvh<T>::value || is_index_bvh<T>::value>
{
};


template <typename T>
struct is_bvh_inst : std::false_type {};

template <typename T>
struct is_bvh_inst<bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_index_bvh_inst : std::false_type {};

template <typename T>
struct is_index_bvh_inst<index_bvh_inst_t<T>> : std::true_type {};

template <typename T>
struct is_any_bvh_inst : std::integral_constant<bool, is_bvh_inst<T>::value || is_index_bvh_inst<T>::value>
{
};

} // visionaray

#endif // VSNRAY_BVH_TRAITS_H
