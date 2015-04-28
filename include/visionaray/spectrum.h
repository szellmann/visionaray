// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SPECTRUM_H
#define VSNRAY_SPECTRUM_H

#include <visionaray/detail/macros.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Spectral color representation
//

template <typename T>
using spectrum      = vector<3, T>; // TODO: for now, this is an RGB color


//-------------------------------------------------------------------------------------------------
// Color space conversions
//

template <typename T>
VSNRAY_FUNC
inline T rgb_to_luminance(vector<3, T> const& rgb)
{
    return T(0.3) * rgb.x + T(0.59) * rgb.y + T(0.11) * rgb.z;
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> rgba_to_rgb(vector<4, T> const& rgba)
{
    auto inv = select( rgba.w != T(0.0), T(1.0) / rgba.w, T(1.0) );
    return vector<3, T>( rgba.x * inv, rgba.y * inv, rgba.z * inv );
}

template <typename T>
VSNRAY_FUNC
inline vector<4, T> rgb_to_rgba(vector<3, T> const& rgb)
{
    return vector<4, T>( rgb, T(1.0) );
}

} // visionaray

#endif // VSNRAY_SPECTRUM_H
