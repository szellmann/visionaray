// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_FORWARD_H
#define VSNRAY_TEXTURE_FORWARD_H 1

#include <cstddef>

namespace visionaray
{

//--------------------------------------------------------------------------------------------------
// Declarations
//

enum tex_address_mode
{
    Wrap = 0,
    Mirror,
    Clamp,
    Border
};


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol,
    CardinalSpline
};


template <typename T, size_t Dim>
class texture_base;

template <typename T, size_t Dim>
class texture_ref_base;

template <typename Base, typename T, size_t Dim>
class texture_iface;

template <typename T, size_t Dim>
using texture = texture_iface<texture_base<T, Dim>, T, Dim>;

template <typename T, size_t Dim>
using texture_ref = texture_iface<texture_ref_base<T, Dim>, T, Dim>;

} // visionaray

#endif // VSNRAY_TEXTURE_FORWARD_H
