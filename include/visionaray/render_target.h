// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_TARGET_H
#define VSNRAY_RENDER_TARGET_H

#include <cstdint>
#include <vector>

#include "detail/aligned_vector.h"
#include "pixel_traits.h"

namespace visionaray
{

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

template <pixel_format ColorFormat, pixel_format DepthFormat>
class cpu_buffer_rt : public render_target
{
public:

    using color_traits  = pixel_traits<ColorFormat>;
    using depth_traits  = pixel_traits<DepthFormat>;
    using color_type    = typename color_traits::type;
    using depth_type    = typename depth_traits::type;

public:

    color_type* color();
    depth_type* depth();

    color_type const* color() const;
    depth_type const* depth() const;

    void begin_frame();
    void end_frame();
    void resize(size_t w, size_t h);
    void display_color_buffer() const;

private:

    aligned_vector<color_type> color_buffer_;
    aligned_vector<depth_type> depth_buffer_;

};

} // visionaray

#include "detail/render_target.inl"

#endif // VSNRAY_RENDER_TARGET_H
