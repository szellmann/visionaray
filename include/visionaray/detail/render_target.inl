// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef NDEBUG
#include <iostream>
#endif

#include <visionaray/gl/util.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::color_type* cpu_buffer_rt<CF, DF>::color()
{
    return color_buffer_.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::depth_type* cpu_buffer_rt<CF, DF>::depth()
{
    return depth_buffer_.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::color_type const* cpu_buffer_rt<CF, DF>::color() const
{
    return color_buffer_.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::depth_type const* cpu_buffer_rt<CF, DF>::depth() const
{
    return depth_buffer_.data();
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::begin_frame()
{
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::end_frame()
{
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::resize(size_t w, size_t h)
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
void cpu_buffer_rt<CF, DF>::display_color_buffer() const
{

    pixel_format_info info = map_pixel_format(color_traits::format);
    gl::blend_pixels( width(), height(), info.format, info.type, color_buffer_.data() );

}

} // visionaray
