// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iostream>
#endif

#include "gl/util.h"
#include "render_target.h"

namespace visionaray
{


//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

cpu_buffer_rt::cpu_buffer_rt(pixel_format cf, pixel_format df)
    : color_format_(cf)
    , depth_format_(df)
{
}

void* cpu_buffer_rt::color()
{
    return reinterpret_cast<void*>( color_buffer_.data() );
}

void* cpu_buffer_rt::depth()
{
    return reinterpret_cast<void*>( depth_buffer_.data() );
}

void const* cpu_buffer_rt::color() const
{
    return reinterpret_cast<void const*>( color_buffer_.data() );
}

void const* cpu_buffer_rt::depth() const
{
    return reinterpret_cast<void const*>( depth_buffer_.data() );
}

void cpu_buffer_rt::begin_frame_impl()
{
}

void cpu_buffer_rt::end_frame_impl()
{
}

void cpu_buffer_rt::resize_impl(size_t w, size_t h)
{

    pixel_format_info cinfo = map_pixel_format(color_format_);
    color_buffer_.resize( w * h * cinfo.size );

    if (depth_format_ != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_format_);
        depth_buffer_.resize( w * h * dinfo.size );
    }
}

void cpu_buffer_rt::display_color_buffer_impl() const
{

    pixel_format_info info = map_pixel_format(color_format_);
    gl::blend_pixels( width(), height(), info.format, info.type, color_buffer_.data() );

}

} // visionaray


