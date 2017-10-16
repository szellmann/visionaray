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

template <pixel_format ColorFormat, pixel_format DepthFormat = PF_UNSPECIFIED>
struct render_target_ref
{
    using color_type = typename pixel_traits<ColorFormat>::type;
    using depth_type = typename pixel_traits<DepthFormat>::type;

    VSNRAY_FUNC color_type* color()
    {
        return color_;
    }

    VSNRAY_FUNC depth_type* depth()
    {
        return depth_;
    }

    VSNRAY_FUNC color_type const* color() const
    {
        return color_;
    }

    VSNRAY_FUNC depth_type const* depth() const
    {
        return depth_;
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

    int width_;
    int height_;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H
