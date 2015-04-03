// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GPU_BUFFER_RT_H
#define VSNRAY_GPU_BUFFER_RT_H

#include <thrust/copy.h>
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

    color_type* color()
    {
        return thrust::raw_pointer_cast(color_buffer_.data());
    }

    depth_type* depth()
    {
        return thrust::raw_pointer_cast(depth_buffer_.data());
    }

    color_type const* color() const
    {
        return thrust::raw_pointer_cast(color_buffer_.data());
    }

    depth_type const* depth() const
    {
        return thrust::raw_pointer_cast(depth_buffer_.data());
    }

    void begin_frame() {}
    void end_frame() {}

    void resize(size_t w, size_t h)
    {
        pixel_format_info cinfo = map_pixel_format(color_traits::format);
        color_buffer_.resize( w * h * cinfo.size );

        if (depth_traits::format != PF_UNSPECIFIED)
        {
            pixel_format_info dinfo = map_pixel_format(depth_traits::format);
            depth_buffer_.resize( w * h * dinfo.size );
        }

        render_target::resize(w, h);
    }

    void display_color_buffer() const
    {
        cpu_buffer_rt<ColorFormat, DepthFormat> rt = *this;
        rt.display_color_buffer();
    }

    operator cpu_buffer_rt<ColorFormat, DepthFormat>() const
    {
        cpu_buffer_rt<ColorFormat, DepthFormat> rt;

        rt.resize( width(), height() );

        // TODO: make render targets templates!
        // This won't compile if cpu_buffer_rt::XXX_traits::format
        //      != gpu_buffer_rt::XXX_traits::format
        thrust::copy( color_buffer_.begin(), color_buffer_.end(), rt.color() );

        if (depth_traits::format != PF_UNSPECIFIED)
        {
            thrust::copy( depth_buffer_.begin(), depth_buffer_.end(), rt.depth() );
        }

        return rt;
    }

private:

    thrust::device_vector<color_type> color_buffer_;
    thrust::device_vector<depth_type> depth_buffer_;

};


} // visionaray

#endif // VSNRAY_GPU_BUFFER_RT_H
