// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_COLOR_CONVERSION_H
#define VSNRAY_DETAIL_COLOR_CONVERSION_H

#include <visionaray/math/math.h>

#include "macros.h"


namespace visionaray
{


template <typename TargetType, typename SourceType>
VSNRAY_FUNC
inline void convert(TargetType& target, SourceType const& source)
{
    target = static_cast<TargetType>(source);
}


// RGBA8 <-- RGBA32F
VSNRAY_FUNC
inline void convert(vector<4, unorm<8>>& target, vec4 const& source)
{
    target = vector<4, unorm<8>>(clamp(source, vec4(0.0), vec4(1.0)));
}

// RGBA32F <-- RGBA8
// Ok.


} // visionaray


#endif // VSNRAY_DETAIL_COLOR_CONVERSION_H
