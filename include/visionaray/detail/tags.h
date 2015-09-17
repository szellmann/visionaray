// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TAGS_H
#define VSNRAY_DETAIL_TAGS_H


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
struct has_textures_tag : surface_tag {};
struct has_no_textures_tag : surface_tag {};

template <typename T>
struct has_textures_impl
{
    template <typename U>
    static has_textures_tag  test(typename U::has_textures*);

    template <typename U>
    static has_no_textures_tag test(...);

    using type = decltype( test<typename std::decay<T>::type>(nullptr) );
};

template <class T>
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
