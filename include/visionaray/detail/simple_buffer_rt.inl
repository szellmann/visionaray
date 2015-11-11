// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Accessors
//

template <pixel_format CF, pixel_format DF>
typename simple_buffer_rt<CF, DF>::color_type* simple_buffer_rt<CF, DF>::color()
{
    return color_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename simple_buffer_rt<CF, DF>::depth_type* simple_buffer_rt<CF, DF>::depth()
{
    return depth_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename simple_buffer_rt<CF, DF>::color_type const* simple_buffer_rt<CF, DF>::color() const
{
    return color_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename simple_buffer_rt<CF, DF>::depth_type const* simple_buffer_rt<CF, DF>::depth() const
{
    return depth_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename simple_buffer_rt<CF, DF>::ref_type simple_buffer_rt<CF, DF>::ref()
{
    return typename simple_buffer_rt<CF, DF>::ref_type( color(), depth() );
}


//-------------------------------------------------------------------------------------------------
// Interface
//

template <pixel_format CF, pixel_format DF>
void simple_buffer_rt<CF, DF>::begin_frame()
{
}

template <pixel_format CF, pixel_format DF>
void simple_buffer_rt<CF, DF>::end_frame()
{
}

template <pixel_format CF, pixel_format DF>
void simple_buffer_rt<CF, DF>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);


    pixel_format_info cinfo = map_pixel_format(color_traits::format);
    color_buffer.resize( w * h * cinfo.size );

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format);
        depth_buffer.resize( w * h * dinfo.size );
    }
}

} // visionaray
