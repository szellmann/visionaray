// This file is distributed under the MIT license.
// See the LICENSE file for details.

#if defined(__CUDACC__)
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#elif defined(__HCC__)
#include "../hcc/algorithm.h"
#endif

#include "../cpu_buffer_rt.h"

namespace visionaray
{

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type* gpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
#if defined(__CUDACC__)
    return thrust::raw_pointer_cast(color_buffer_.data());
#elif defined(__HCC__)
    return color_buffer_.data();
#endif
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* gpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
#if defined(__CUDACC__)
    return thrust::raw_pointer_cast(depth_buffer_.data());
#elif defined(__HCC__)
    return depth_buffer_.data();
#endif
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
#if defined(__CUDACC__)
    return thrust::raw_pointer_cast(color_buffer_.data());
#elif defined(__HCC__)
    return color_buffer_.data();
#endif
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
#if defined(__CUDACC__)
    return thrust::raw_pointer_cast(depth_buffer_.data());
#elif defined(__HCC__)
    return depth_buffer_.data();
#endif
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type gpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
#if defined(__CUDACC__)
    return { color(), depth(), width(), height() };
#elif defined(__HCC__)
    // TODO: check why aggregate initialization doesn't work with hcc here
    typename gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type result;
    result.color_  = color();
    result.depth_  = depth();
    result.width_  = width();
    result.height_ = height();
    return result;
#endif
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

#if defined(__CUDACC__)
    thrust::fill(thrust::device, color(), color() + width() * height(), cc);
#elif defined(__HCC__)
    hcc::fill(hcc::device, color(), color() + width() * height(), cc);
#endif
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

#if defined(__CUDACC__)
    thrust::fill(thrust::device, depth(), depth() + width() * height(), dd);
#elif defined(__HCC__)
    hcc::fill(hcc::device, depth(), depth() + width() * height(), dd);
#endif
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
void gpu_buffer_rt<ColorFormat, DepthFormat>::resize(int w, int h)
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
#if defined(__CUDACC__)
    thrust::copy( color_buffer_.begin(), color_buffer_.end(), rt.color() );
#elif defined(__HCC__)
    pixel_format_info info = map_pixel_format(ColorFormat);

    auto alloc = color_buffer_.get_allocator();
    hc::accelerator_view av = alloc.accelerator().get_default_view();
    av.copy(color_buffer_.data(), rt.color(), color_buffer_.size() * info.size);
#endif

    if (DepthFormat != PF_UNSPECIFIED)
    {
#if defined(__CUDACC__)
        thrust::copy( depth_buffer_.begin(), depth_buffer_.end(), rt.depth() );
#elif defined(__HCC__)
        pixel_format_info info = map_pixel_format(DepthFormat);

        auto alloc = depth_buffer_.get_allocator();
        hc::accelerator_view av = alloc.accelerator().get_default_view();
        av.copy(depth_buffer_.data(), rt.depth(), depth_buffer_.size() * info.size);
#endif
    }

    rt.display_color_buffer();
}

} // visionaray
