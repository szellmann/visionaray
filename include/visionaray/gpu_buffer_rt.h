// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GPU_BUFFER_RT_H
#define VSNRAY_GPU_BUFFER_RT_H

#include <thrust/device_vector.h>

#include "render_target.h"

namespace visionaray
{

// TODO: have a *single* buffered render target template
// from either std::vector or thrust::device_vector???
template <pixel_format ColorFormat, pixel_format DepthFormat>
class gpu_buffer_rt : public render_target
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

    operator cpu_buffer_rt<ColorFormat, DepthFormat>() const;

private:

    thrust::device_vector<color_type> color_buffer_;
    thrust::device_vector<depth_type> depth_buffer_;

};


} // visionaray

#endif // VSNRAY_GPU_BUFFER_RT_H
