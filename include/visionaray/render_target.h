// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_TARGET_H
#define VSNRAY_RENDER_TARGET_H

#include <cstddef>

#include "pixel_traits.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Render target base
//

class render_target
{
public:

    void resize(size_t w, size_t h)
    {
        width_  = w;
        height_ = h;
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

private:

    size_t width_;
    size_t height_;

};


//-------------------------------------------------------------------------------------------------
// Render target ref
//

template <pixel_format ColorFormat, pixel_format DepthFormat = PF_UNSPECIFIED>
class render_target_ref
{
public:

    using color_type = typename pixel_traits<ColorFormat>::type;
    using depth_type = typename pixel_traits<DepthFormat>::type;

public:

    render_target_ref(color_type* color, depth_type* depth)
        : color_(color)
        , depth_(depth)
    {
    }

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

private:

    color_type* color_ = nullptr;
    depth_type* depth_ = nullptr;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H
