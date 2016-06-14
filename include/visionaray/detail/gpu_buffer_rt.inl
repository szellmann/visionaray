// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <thrust/copy.h>

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
    pixel_format_info cinfo = map_pixel_format(ColorFormat);
    color_buffer_.resize( w * h * cinfo.size );

    if (DepthFormat != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(DepthFormat);
        depth_buffer_.resize( w * h * dinfo.size );
    }

    render_target::resize(w, h);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    cpu_buffer_rt<ColorFormat, DepthFormat> rt = *this;
    rt.display_color_buffer();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
gpu_buffer_rt<ColorFormat, DepthFormat>::operator cpu_buffer_rt<ColorFormat, DepthFormat>() const
{
    cpu_buffer_rt<ColorFormat, DepthFormat> rt;

    rt.resize( width(), height() );

    // TODO: make render targets templates!
    // This won't compile if cpu_buffer_rt::XXX_traits::format
    //      != gpu_buffer_rt::XXX_traits::format
    thrust::copy( color_buffer_.begin(), color_buffer_.end(), rt.color() );

    if (DepthFormat != PF_UNSPECIFIED)
    {
        thrust::copy( depth_buffer_.begin(), depth_buffer_.end(), rt.depth() );
    }

    return rt;
}

} // visionaray
