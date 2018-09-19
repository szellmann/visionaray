// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE_TRAITS_H
#define VSNRAY_TEXTURE_TEXTURE_TRAITS_H 1

namespace visionaray
{

template <typename T>
struct texture_dimensions
{
    enum { value = T::dimensions };
};

} // visionaray

#endif // VSNRAY_TEXTURE_TEXTURE_TRAITS_H
