// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CPU_BUFFER_RT_H
#define VSNRAY_CPU_BUFFER_RT_H 1

#include <memory>

#include "pixel_traits.h"
#include "render_target.h"

namespace visionaray
{

template <pixel_format ColorFormat, pixel_format DepthFormat>
class cpu_buffer_rt : public render_target
{
public:

    using color_type    = typename pixel_traits<ColorFormat>::type;
    using depth_type    = typename pixel_traits<DepthFormat>::type;

    using ref_type      = render_target_ref<ColorFormat, DepthFormat>;

public:

    cpu_buffer_rt();
   ~cpu_buffer_rt();

    color_type* color();
    depth_type* depth();

    color_type const* color() const;
    depth_type const* depth() const;

    ref_type ref();

    void clear_color(color_type color = color_type(0.0));
    void clear_depth(depth_type depth = depth_type(1.0));
    void begin_frame();
    void end_frame();
    void resize(size_t w, size_t h);
    void display_color_buffer() const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

} // visionaray

#include "detail/cpu_buffer_rt.inl"

#endif // VSNRAY_CPU_BUFFER_RT_H
