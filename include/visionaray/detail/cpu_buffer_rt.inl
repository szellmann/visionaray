// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include "color_conversion.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
cpu_buffer_rt<ColorFormat, DepthFormat>::cpu_buffer_rt()
    : compositor(nullptr)
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
cpu_buffer_rt<ColorFormat, DepthFormat>::~cpu_buffer_rt() = default;

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::color_type* cpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* cpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* cpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* cpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::ref_type cpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return { color(), depth(), width(), height() };
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::clear_color_buffer(vec4 const& c)
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

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::clear_depth_buffer(float d)
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

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::resize(int w, int h)
{
    render_target::resize(w, h);


    // Allocate storage

    color_buffer.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        depth_buffer.resize(w * h);
    }

    if (!compositor)
    {
        compositor.reset(new gl::depth_compositor);
    }

    // Allocate texture storage

    pixel_format_info cinfo = map_pixel_format(ColorFormat);

    compositor->setup_color_texture(cinfo, w, h);


    if (DepthFormat != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        compositor->setup_depth_texture(dinfo, w, h);
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    if (DepthFormat != PF_UNSPECIFIED)
    {
        // Update color texture

        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                color_buffer.data()
                );


        // Update depth texture

        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        compositor->update_depth_texture(
                dinfo,
                width(),
                height(),
                depth_buffer.data()
                );


        // Combine textures using a shader

        compositor->composite_textures();
    }
    else
    {
        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                color_buffer.data()
                );

        compositor->display_color_texture();
    }
}

} // visionaray
