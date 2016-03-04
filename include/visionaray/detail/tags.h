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

struct surface_tag {};

struct has_normals_tag      : surface_tag {};
struct has_no_normals_tag   : surface_tag {};

struct has_colors_tag       : surface_tag {};
struct has_no_colors_tag    : surface_tag {};

struct has_textures_tag     : surface_tag {};
struct has_no_textures_tag  : surface_tag {};

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
// Misc.
//

struct have_intersector_tag {};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_TAGS_H
