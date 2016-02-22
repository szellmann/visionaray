// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <thrust/copy.h>

namespace visionaray
{

template <pixel_format CF, pixel_format DF>
typename gpu_buffer_rt<CF, DF>::color_type* gpu_buffer_rt<CF, DF>::color()
{
    return thrust::raw_pointer_cast(color_buffer_.data());
}

template <pixel_format CF, pixel_format DF>
typename gpu_buffer_rt<CF, DF>::depth_type* gpu_buffer_rt<CF, DF>::depth()
{
    return thrust::raw_pointer_cast(depth_buffer_.data());
}

template <pixel_format CF, pixel_format DF>
typename gpu_buffer_rt<CF, DF>::color_type const* gpu_buffer_rt<CF, DF>::color() const
{
    return thrust::raw_pointer_cast(color_buffer_.data());
}

template <pixel_format CF, pixel_format DF>
typename gpu_buffer_rt<CF, DF>::depth_type const* gpu_buffer_rt<CF, DF>::depth() const
{
    return thrust::raw_pointer_cast(depth_buffer_.data());
}

template <pixel_format CF, pixel_format DF>
typename gpu_buffer_rt<CF, DF>::ref_type gpu_buffer_rt<CF, DF>::ref()
{
    return gpu_buffer_rt<CF, DF>::ref_type( color(), depth() );
}

template <pixel_format CF, pixel_format DF>
void gpu_buffer_rt<CF, DF>::begin_frame()
{
}

template <pixel_format CF, pixel_format DF>
void gpu_buffer_rt<CF, DF>::end_frame()
{
}

template <pixel_format CF, pixel_format DF>
void gpu_buffer_rt<CF, DF>::resize(size_t w, size_t h)
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

template <pixel_format CF, pixel_format DF>
void gpu_buffer_rt<CF, DF>::display_color_buffer() const
{
    cpu_buffer_rt<CF, DF> rt = *this;
    rt.display_color_buffer();
}

template <pixel_format CF, pixel_format DF>
gpu_buffer_rt<CF, DF>::operator cpu_buffer_rt<CF, DF>() const
{
    cpu_buffer_rt<CF, DF> rt;

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

} // visionaray
