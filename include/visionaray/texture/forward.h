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

enum tex_color_space
{
    RGB = 0,
    sRGB
};


template <typename T, size_t Dim>
class cuda_texture;

template <typename T, size_t Dim>
class cuda_texture_ref;

} // visionaray

#endif // VSNRAY_TEXTURE_FORWARD_H
