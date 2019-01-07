// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE_TRAITS_H
#define VSNRAY_TEXTURE_TEXTURE_TRAITS_H 1

#include <type_traits>
#include <utility>

#include <visionaray/math/vector.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Traits for texture types
//
//
//  - is_texture:
//      determine if type is a Texture. The minimum requirement for a type
//      to adhere to the Texture concept is that there exists either:
//
//      * a free function tex1D(T, float)
//      * a free function tex2D(T, vector<2, float>)
//      * a free function tex3D(T, vector<3, float>)
//
//      that returns a point sample
//
//  - texture_dimensions:
//      value can be either of {0,1,2,3}, where 0 denotes that type does
//      not adhere to concept Texture
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// is_texture
//

namespace detail
{

template <typename T>
struct is_texture_impl
{
    template <typename U>
    static constexpr std::true_type test(decltype(tex1D(std::declval<U>(), 0.0f))*);

    template <typename U>
    static constexpr std::true_type test(decltype(tex2D(std::declval<U>(), vector<2, float>(0.0f)))*, void* = 0);

    template <typename U>
    static constexpr std::true_type test(decltype(tex3D(std::declval<U>(), vector<3, float>(0.0f)))*, void* = 0, void* = 0);

    template <typename U>
    static constexpr std::false_type test(...);

    using type = decltype(test<typename std::decay<T>::type>(nullptr));
};

} // detail


template <typename T>
struct is_texture : detail::is_texture_impl<T>::type
{
};


//-------------------------------------------------------------------------------------------------
// texture_dimensions
//

template <typename T, typename Enable = void>
struct texture_dimensions
{
    enum { value = 0 };
};

template <typename T>
struct texture_dimensions<T, typename std::enable_if<is_texture<T>::value>::type>
{
    enum { value = T::dimensions };
};

} // visionaray

#endif // VSNRAY_TEXTURE_TEXTURE_TRAITS_H
