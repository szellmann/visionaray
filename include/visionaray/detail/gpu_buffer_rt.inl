// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "../cpu_buffer_rt.h"

namespace visionaray
{

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type* gpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return thrust::raw_pointer_cast(color_buffer_.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* gpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return thrust::raw_pointer_cast(depth_buffer_.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return thrust::raw_pointer_cast(color_buffer_.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return thrust::raw_pointer_cast(depth_buffer_.data());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type gpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type(
            color(),
            depth(),
            width(),
            height()
            );
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::clear_color_buffer(vec4 const& c)
{
    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<ColorFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    thrust::fill(thrust::device, color(), color() + width() * height(), cc);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::clear_depth_buffer(float d)
{
    // Convert from DEPTH32F to internal depth format
    depth_type dd;
    convert(
        pixel_format_constant<DepthFormat>{},
        pixel_format_constant<PF_DEPTH32F>{},
        dd,
        d
        );

    thrust::fill(thrust::device, depth(), depth() + width() * height(), dd);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);

    color_buffer_.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        depth_buffer_.resize(w * h);
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    cpu_buffer_rt<ColorFormat, DepthFormat> rt;

    rt.resize( width(), height() );

    // TODO:
    // This won't compile if cpu_buffer_rt::XXX_traits::format
    //      != gpu_buffer_rt::XXX_traits::format
    thrust::copy( color_buffer_.begin(), color_buffer_.end(), rt.color() );

    if (DepthFormat != PF_UNSPECIFIED)
    {
        thrust::copy( depth_buffer_.begin(), depth_buffer_.end(), rt.depth() );
    }

    rt.display_color_buffer();
}

} // visionaray
