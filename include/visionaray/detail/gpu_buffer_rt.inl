// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime.h>

#include "../cuda/fill.h"
#include "../cpu_buffer_rt.h"

namespace visionaray
{

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type* gpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return color_buffer_;
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* gpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return depth_buffer_;
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return color_buffer_;
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* gpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return depth_buffer_;
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename gpu_buffer_rt<ColorFormat, DepthFormat>::ref_type gpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return { color(), depth(), width(), height() };
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

    cuda::fill(color(), width() * height() * sizeof(color_type), &cc, sizeof(color_type));
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

    cuda::fill(depth(), width() * height() * sizeof(depth_type), &dd, sizeof(depth_type));
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
    if (w == width() && h == height())
    {
        return;
    }

    render_target::resize(w, h);

    cudaFree(color_buffer_);
    cudaMalloc((void**)&color_buffer_, w * h * sizeof(color_type));

    if (DepthFormat != PF_UNSPECIFIED)
    {
        cudaFree(depth_buffer_);
        cudaMalloc((void**)&depth_buffer_, w * h * sizeof(depth_type));
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void gpu_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    cpu_buffer_rt<ColorFormat, DepthFormat> rt;

    rt.resize(width(), height());

    cudaMemcpy(
        rt.color(),
        color_buffer_,
        width() * height() * sizeof(color_type),
        cudaMemcpyDeviceToHost
        );

    if (DepthFormat != PF_UNSPECIFIED)
    {
        cudaMemcpy(
            rt.depth(),
            depth_buffer_,
            width() * height() * sizeof(depth_type),
            cudaMemcpyDeviceToHost
            );
    }

    rt.display_color_buffer();
}

} // visionaray
