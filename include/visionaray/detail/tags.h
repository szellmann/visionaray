// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_TAGS_H
#define VSNRAY_DETAIL_TAGS_H

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

template <typename T> struct always_void { using type = void; };
template <typename T> using always_void_t = typename always_void<T>::type;

template <typename T, typename = void>
struct has_textures : has_no_textures_tag {};

template <typename T>
struct has_textures<T, always_void_t<decltype(T::textures)>> : has_textures_tag {};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_TAGS_H


