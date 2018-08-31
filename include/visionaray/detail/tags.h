// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TAGS_H
#define VSNRAY_DETAIL_TAGS_H 1

#include <type_traits>


//-------------------------------------------------------------------------------------------------
// Tags for internal use
//

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Surface
//

using has_normals_tag     = std::true_type;
using has_no_normals_tag  = std::false_type;

using has_colors_tag      = std::true_type;
using has_no_colors_tag   = std::false_type;

using has_textures_tag    = std::true_type;
using has_no_textures_tag = std::false_type;

template <typename T>
struct has_normals_impl
{
    template <typename U>
    static has_normals_tag test(typename U::has_normals*);

    template <typename U>
    static has_no_normals_tag test(...);

    using type = decltype( test<typename std::decay<T>::type>(nullptr) );
};

template <typename T>
struct has_normals : has_normals_impl<T>::type
{
};

template <typename T>
struct has_colors_impl
{
    template <typename U>
    static has_colors_tag test(typename U::has_colors*);

    template <typename U>
    static has_no_colors_tag test(...);

    using type = decltype( test<typename std::decay<T>::type>(nullptr) );
};

template <typename T>
struct has_colors : has_colors_impl<T>::type
{
};

template <typename T>
struct has_textures_impl
{
    template <typename U>
    static has_textures_tag test(typename U::has_textures*);

    template <typename U>
    static has_no_textures_tag test(...);

    using type = decltype( test<typename std::decay<T>::type>(nullptr) );
};

template <typename T>
struct has_textures : has_textures_impl<T>::type
{
};


//-------------------------------------------------------------------------------------------------
// Traversal types
//

enum traversal_type { AnyHit, ClosestHit, MultiHit };

using any_hit_tag     = std::integral_constant<int, AnyHit>;
using closest_hit_tag = std::integral_constant<int, ClosestHit>;
using multi_hit_tag   = std::integral_constant<int, MultiHit>;


//-------------------------------------------------------------------------------------------------
// Misc.
//

struct have_intersector_tag {};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_TAGS_H
