// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_COLOR_CONVERSION_H
#define VSNRAY_DETAIL_COLOR_CONVERSION_H

#include <cstdint>

#include <visionaray/math/math.h>
#include <visionaray/norm.h>
#include <visionaray/pixel_format.h>

#include "macros.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Color conversion functors
//

template <pixel_format PF>
struct convert
{
};

template <>
struct convert<PF_RGB8>
{
    typedef vector<3, unorm<8>> internal_type;

    VSNRAY_FUNC
    inline internal_type operator()(vec3 const& color) const
    {
        return internal_type(color);
    }

    VSNRAY_FUNC
    inline vec3 operator()(internal_type const& color) const
    {
        return vec3(color);
    }
};

template <>
struct convert<PF_RGBA8>
{
    typedef vector<4, unorm<8>> internal_type;

    VSNRAY_FUNC
    inline internal_type operator()(vec4 const& color) const
    {
        return internal_type(clamp(color, vec4(0.0), vec4(1.0)));
    }

    VSNRAY_FUNC
    inline vec4 operator()(internal_type const& color) const
    {
        return vec4(color);
    }
};

template <>
struct convert<PF_RGBA32F>
{
    VSNRAY_FUNC
    inline vec4 operator()(vec4 const& color) const
    {
        return vec4(color);
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_COLOR_CONVERSION_H
