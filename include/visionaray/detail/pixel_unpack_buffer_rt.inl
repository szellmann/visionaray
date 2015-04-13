// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdexcept>

#include <GL/glew.h>

#include <cuda_gl_interop.h>

#include <visionaray/cuda/graphics_resource.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/util.h>


namespace visionaray
{

template <pixel_format CF, pixel_format DF>
struct pixel_unpack_buffer_rt<CF, DF>::impl
{
    cuda::graphics_resource resource;
    gl::buffer              buffer;
    gl::texture             texture;
};

template <pixel_format CF, pixel_format DF>
pixel_unpack_buffer_rt<CF, DF>::pixel_unpack_buffer_rt()
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
    if (err != cudaSuccess)
    {
        throw std::runtime_error("set GL device");
    }
}

template <pixel_format CF, pixel_format DF>
pixel_unpack_buffer_rt<CF, DF>::~pixel_unpack_buffer_rt()
{
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::color_type* pixel_unpack_buffer_rt<CF, DF>::color()
{
    return static_cast<typename pixel_unpack_buffer_rt<CF, DF>::color_type*>(impl_->resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::depth_type* pixel_unpack_buffer_rt<CF, DF>::depth()
{
    return nullptr;
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::color_type const* pixel_unpack_buffer_rt<CF, DF>::color() const
{
    return static_cast<pixel_unpack_buffer_rt<CF, DF>::color_type const*>(impl_->resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::depth_type const* pixel_unpack_buffer_rt<CF, DF>::depth() const
{
    return nullptr;
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::ref_type pixel_unpack_buffer_rt<CF, DF>::ref()
{
    return typename pixel_unpack_buffer_rt<CF, DF>::ref_type( color(), depth() );
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::begin_frame()
{
    if (impl_->resource.map() == 0)
    {
        throw std::runtime_error("bad resource mapped");
    }
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::end_frame()
{
    pixel_format_info cinfo = map_pixel_format(color_traits::format);

    impl_->resource.unmap();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, impl_->texture.get());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->buffer.get());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), cinfo.format, cinfo.type, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::resize(size_t w, size_t h)
{
    pixel_format_info cinfo = map_pixel_format(color_traits::format);

    // gl texture
    impl_->texture.reset( gl::create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->texture.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, cinfo.internal_format, w, h, 0, cinfo.format, cinfo.type, 0);

    // gl buffer
    impl_->buffer.reset( gl::create_buffer() );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * cinfo.size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register buffer object with CUDA
    impl_->resource.register_buffer(impl_->buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard);

    render_target::resize(w, h);
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::display_color_buffer() const
{
    gl::blend_texture(impl_->texture.get());
}

} // visionaray
