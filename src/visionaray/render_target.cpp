// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iostream>
#endif

#include "gl/util.h"
#include "render_target.h"

namespace visionaray
{

using color_type = cpu_buffer_rt::color_type;
using depth_type = cpu_buffer_rt::depth_type;

//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

color_type* cpu_buffer_rt::color()
{
    return color_buffer_.data();
}

depth_type* cpu_buffer_rt::depth()
{
    return depth_buffer_.data();
}

color_type const* cpu_buffer_rt::color() const
{
    return color_buffer_.data();
}

depth_type const* cpu_buffer_rt::depth() const
{
    return depth_buffer_.data();
}

void cpu_buffer_rt::begin_frame_impl()
{
}

void cpu_buffer_rt::end_frame_impl()
{
}

void cpu_buffer_rt::resize_impl(size_t w, size_t h)
{

    pixel_format_info cinfo = map_pixel_format(color_traits::format());
    color_buffer_.resize( w * h * cinfo.size );

    if (depth_traits::format() != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format());
        depth_buffer_.resize( w * h * dinfo.size );
    }
}

void cpu_buffer_rt::display_color_buffer_impl() const
{

    pixel_format_info info = map_pixel_format(color_traits::format());
    gl::blend_pixels( width(), height(), info.format, info.type, color_buffer_.data() );

}

} // visionaray


