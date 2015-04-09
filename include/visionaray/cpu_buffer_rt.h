// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CPU_BUFFER_RT_H
#define VSNRAY_CPU_BUFFER_RT_H

#include "detail/aligned_vector.h"
#include "pixel_traits.h"
#include "render_target.h"

namespace visionaray
{

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

#include "detail/cpu_buffer_rt.inl"

#endif // VSNRAY_CPU_BUFFER_RT_H
