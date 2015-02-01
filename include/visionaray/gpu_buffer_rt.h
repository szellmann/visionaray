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
class gpu_buffer_rt : public render_target
{
public:

    typedef thrust::device_vector<uint8_t> buffer_type;

    gpu_buffer_rt(pixel_format cf, pixel_format df)
        : color_format_(cf)
        , depth_format_(df)
    {
    }

    void* color()
    {
        return static_cast<void*>( thrust::raw_pointer_cast(color_buffer_.data()) );
    }

    void* depth()
    {
        return static_cast<void*>( thrust::raw_pointer_cast(depth_buffer_.data()) );
    }

    void const* color() const
    {
        return static_cast<void const*>( thrust::raw_pointer_cast(color_buffer_.data()) );
    }

    void const* depth() const
    {
        return static_cast<void const*>( thrust::raw_pointer_cast(depth_buffer_.data()) );
    }

    operator cpu_buffer_rt() const
    {
        cpu_buffer_rt rt(color_format_, depth_format_);
        rt.resize( width(), height() );

        thrust::copy( color_buffer_.begin(), color_buffer_.end(), static_cast<uint8_t*>(rt.color()) );

        if (depth_format_ != PF_UNSPECIFIED)
        {
            thrust::copy( depth_buffer_.begin(), depth_buffer_.end(), static_cast<uint8_t*>(rt.depth()) );
        }

        return rt;
    }

private:

    VSNRAY_NOT_COPYABLE(gpu_buffer_rt)

    buffer_type color_buffer_;
    buffer_type depth_buffer_;

    pixel_format color_format_;
    pixel_format depth_format_;

    void begin_frame_impl() {}
    void end_frame_impl() {}

    void resize_impl(size_t w, size_t h)
    {
        pixel_format_info cinfo = map_pixel_format(color_format_);
        color_buffer_.resize( w * h * cinfo.size );

        if (depth_format_ != PF_UNSPECIFIED)
        {
            pixel_format_info dinfo = map_pixel_format(depth_format_);
            depth_buffer_.resize( w * h * dinfo.size );
        }
    }

    void display_color_buffer_impl() const
    {
        cpu_buffer_rt rt = *this;
        rt.display_color_buffer();
    }

};


} // visionaray

#endif // VSNRAY_GPU_BUFFER_RT_H


