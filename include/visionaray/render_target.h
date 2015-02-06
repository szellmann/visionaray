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

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    void begin_frame()
    {
        begin_frame_impl();
    }

    void end_frame()
    {
        end_frame_impl();
    }

    void resize(size_t w, size_t h)
    {
        width_ = w;
        height_ = h;
        resize_impl(w, h);
    }

    void display_color_buffer() const
    {
        display_color_buffer_impl();
    }

private:

    size_t width_;
    size_t height_;

    virtual void begin_frame_impl() = 0;
    virtual void end_frame_impl() = 0;
    virtual void resize_impl(size_t w, size_t h) = 0;
    virtual void display_color_buffer_impl() const = 0;

};

class cpu_buffer_rt : public render_target
{
public:

    typedef pixel_traits<PF_RGBA32F>        color_traits;
    typedef pixel_traits<PF_UNSPECIFIED>    depth_traits;
    typedef color_traits::type              color_type;
    typedef depth_traits::type              depth_type;

    color_type* color();
    depth_type* depth();

    color_type const* color() const;
    depth_type const* depth() const;

private:

    aligned_vector<color_type> color_buffer_;
    aligned_vector<depth_type> depth_buffer_;

    void begin_frame_impl();
    void end_frame_impl();
    void resize_impl(size_t w, size_t h);
    void display_color_buffer_impl() const;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H


