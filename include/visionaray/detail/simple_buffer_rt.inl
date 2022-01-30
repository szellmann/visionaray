// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Accessors
//

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::color_type* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::color()
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::depth_type* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::depth()
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::accum_type* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::accum()
{
    return accum_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::color_type const* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::color() const
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::depth_type const* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::depth() const
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::accum_type const* simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::accum() const
{
    return accum_buffer.data();
}


//-------------------------------------------------------------------------------------------------
// Interface
//

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
typename simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::ref_type simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::ref()
{
    return { color(), depth(), accum(), width(), height() };
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::clear_color_buffer(vec4 const& c)
{
    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<ColorFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    std::fill(color_buffer.begin(), color_buffer.end(), cc);
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::clear_depth_buffer(float d)
{
    // Convert from DEPTH32F to internal depth format
    depth_type dd;
    convert(
        pixel_format_constant<DepthFormat>{},
        pixel_format_constant<PF_DEPTH32F>{},
        dd,
        d
        );

    std::fill(depth_buffer.begin(), depth_buffer.end(), dd);
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::clear_accum_buffer(vec4 const& c)
{
    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<AccumFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    std::fill(accum_buffer.begin(), accum_buffer.end(), cc);
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat, pixel_format AccumFormat>
void simple_buffer_rt<ColorFormat, DepthFormat, AccumFormat>::resize(int w, int h)
{
    render_target::resize(w, h);


    color_buffer.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        depth_buffer.resize(w * h);
    }

    if (AccumFormat != PF_UNSPECIFIED)
    {
        accum_buffer.resize(w * h);
    }
}

} // visionaray
