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
// Color space conversions
//

template <typename T>
VSNRAY_FUNC
inline T rgb_to_luminance(vector<3, T> const& rgb)
{
    return T(0.3) * rgb.x + T(0.59) * rgb.y + T(0.11) * rgb.z;
}

} // visionaray

#endif // VSNRAY_SPECTRUM_H
