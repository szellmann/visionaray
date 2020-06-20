// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <thrust/fill.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Accessors
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_gpu_buffer_rt<ColorFormat, DepthFormat>::color_type* simple_gpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return thrust::raw_pointer_cast(color_buffer.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* simple_gpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return thrust::raw_pointer_cast(depth_buffer.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_gpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* simple_gpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return thrust::raw_pointer_cast(color_buffer.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* simple_gpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return thrust::raw_pointer_cast(depth_buffer.data());
}


//-------------------------------------------------------------------------------------------------
// Interface
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename simple_gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type simple_gpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return { color(), depth(), width(), height() };
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_gpu_buffer_rt<ColorFormat, DepthFormat>::clear_color_buffer(vec4 const& c)
{
    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<ColorFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    thrust::fill(color_buffer.begin(), color_buffer.end(), cc);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_gpu_buffer_rt<ColorFormat, DepthFormat>::clear_depth_buffer(float d)
{
    // Convert from DEPTH32F to internal depth format
    depth_type dd;
    convert(
        pixel_format_constant<DepthFormat>{},
        pixel_format_constant<PF_DEPTH32F>{},
        dd,
        d
        );

    thrust::fill(depth_buffer.begin(), depth_buffer.end(), dd);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_gpu_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_gpu_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void simple_gpu_buffer_rt<ColorFormat, DepthFormat>::resize(int w, int h)
{
    render_target::resize(w, h);


    color_buffer.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        depth_buffer.resize(w * h);
    }
}

} // visionaray
