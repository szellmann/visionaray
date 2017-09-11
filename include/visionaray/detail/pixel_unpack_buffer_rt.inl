// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <cassert>
#include <stdexcept>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#endif

#include <cuda_gl_interop.h>

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <visionaray/cuda/graphics_resource.h>
#include <visionaray/gl/compositing.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/util.h>

#include "color_conversion.h"


namespace visionaray
{

template <pixel_format ColorFormat, pixel_format DepthFormat>
struct pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::impl
{
    impl() : compositor(nullptr) {}

    std::unique_ptr<gl::depth_compositor>   compositor;

    cuda::graphics_resource                 color_resource;
    cuda::graphics_resource                 depth_resource;

    gl::buffer                              color_buffer;
    gl::buffer                              depth_buffer;
};

template <pixel_format ColorFormat, pixel_format DepthFormat>
pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::pixel_unpack_buffer_rt()
    : impl_(new impl())
{
    cudaError_t err = cudaSuccess;

    int dev = 0;
    cudaDeviceProp prop;
    err = cudaChooseDevice(&dev, &prop);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("choose device");
    }

    err = cudaGLSetGLDevice(dev);

    if (err == cudaErrorSetOnActiveProcess)
    {
        err = cudaDeviceReset();
        err = cudaGLSetGLDevice(dev);
    }

    if (err != cudaSuccess)
    {
        throw std::runtime_error("set GL device");
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::~pixel_unpack_buffer_rt()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::color_type* pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return static_cast<color_type*>(impl_->color_resource.dev_ptr());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::depth_type* pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return static_cast<depth_type*>(impl_->depth_resource.dev_ptr());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::color_type const* pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return static_cast<color_type const*>(impl_->color_resource.dev_ptr());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::depth_type const* pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return static_cast<depth_type const*>(impl_->depth_resource.dev_ptr());
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::ref_type pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return { color(), depth(), width(), height() };
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::clear_color_buffer(vec4 const& c)
{
    assert(color() == 0 && "clear_color_buffer() called between begin_frame() and end_frame()");

    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<ColorFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    begin_frame();

    thrust::fill(thrust::device, color(), color() + width() * height(), cc);

    end_frame();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::clear_depth_buffer(float d)
{
    assert(depth() == 0 && "clear_depth_buffer() called between begin_frame() and end_frame()");

    // Convert from DEPTH32F to internal depth format
    depth_type dd;
    convert(
        pixel_format_constant<DepthFormat>{},
        pixel_format_constant<PF_DEPTH32F>{},
        dd,
        d
        );

    begin_frame();

    thrust::fill(thrust::device, depth(), depth() + width() * height(), dd);

    end_frame();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
    if (impl_->color_resource.map() == 0)
    {
        throw std::runtime_error("bad color resource mapped");
    }

    if (DepthFormat != PF_UNSPECIFIED && impl_->depth_resource.map() == 0)
    {
        throw std::runtime_error("bad depth resource mapped");
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
    impl_->color_resource.unmap();

    if (DepthFormat != PF_UNSPECIFIED)
    {
        impl_->depth_resource.unmap();
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);

    if (!impl_->compositor)
    {
        impl_->compositor.reset(new gl::depth_compositor);
    }

    pixel_format_info cinfo = map_pixel_format(ColorFormat);

    // GL texture
    impl_->compositor->setup_color_texture(cinfo, w, h);

    // GL buffer
    impl_->color_buffer.reset( gl::create_buffer() );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->color_buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * cinfo.size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register buffer object with CUDA
    impl_->color_resource.register_buffer(impl_->color_buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        // GL texture
        impl_->compositor->setup_depth_texture(dinfo, w, h);

        // GL buffer
        impl_->depth_buffer.reset( gl::create_buffer() );

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->depth_buffer.get());
        glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * dinfo.size, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // register buffer object with CUDA
        impl_->depth_resource.register_buffer(impl_->depth_buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard);
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void pixel_unpack_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    if (DepthFormat != PF_UNSPECIFIED)
    {
        // Update color texture

        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->color_buffer.get());

        impl_->compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                0
                );

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        // Update depth texture

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->depth_buffer.get());

        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        impl_->compositor->update_depth_texture(
                dinfo,
                width(),
                height(),
                0
                );

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        // Combine textures using a shader

        impl_->compositor->composite_textures();
    }
    else
    {
        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->color_buffer.get());

        impl_->compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                0
                );

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        impl_->compositor->display_color_texture();
    }
}

} // visionaray
