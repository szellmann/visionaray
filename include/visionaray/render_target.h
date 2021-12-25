// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_TARGET_H
#define VSNRAY_RENDER_TARGET_H 1

#include "detail/macros.h"
#include "pixel_traits.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Render target base
//

class render_target
{
public:

    void resize(int w, int h)
    {
        width_  = w;
        height_ = h;
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:

    int width_;
    int height_;

};


//-------------------------------------------------------------------------------------------------
// Render target ref
//

template <
    pixel_format ColorFormat,
    pixel_format DepthFormat = PF_UNSPECIFIED,
    pixel_format AccumFormat = PF_UNSPECIFIED
    >
struct render_target_ref
{
    constexpr static pixel_format color_format = ColorFormat;
    constexpr static pixel_format depth_format = DepthFormat;
    constexpr static pixel_format accum_format = AccumFormat;

    // Storage type used by the color buffer
    using color_type = typename pixel_traits<ColorFormat>::type;

    // Storage type used by the depth buffer
    using depth_type = typename pixel_traits<DepthFormat>::type;

    // Storage type used by the accumulation buffer
    using accum_type = typename pixel_traits<AccumFormat>::type;


    VSNRAY_FUNC color_type* color()
    {
        return color_;
    }

    VSNRAY_FUNC depth_type* depth()
    {
        return depth_;
    }

    VSNRAY_FUNC accum_type* accum()
    {
        return accum_;
    }

    VSNRAY_FUNC color_type const* color() const
    {
        return color_;
    }

    VSNRAY_FUNC depth_type const* depth() const
    {
        return depth_;
    }

    VSNRAY_FUNC accum_type const* accum() const
    {
        return accum_;
    }

    VSNRAY_FUNC int width() const
    {
        return width_;
    }

    VSNRAY_FUNC int height() const
    {
        return height_;
    }

    // Public, to allow for aggregate initialization!
    color_type* color_;
    depth_type* depth_;
    accum_type* accum_;

    int width_;
    int height_;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H
