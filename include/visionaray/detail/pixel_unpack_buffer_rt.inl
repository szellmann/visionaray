// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <stdexcept>

#include <GL/glew.h>

#include <cuda_gl_interop.h>

#include <visionaray/cuda/graphics_resource.h>
#include <visionaray/gl/compositing.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/util.h>


namespace visionaray
{

template <pixel_format CF, pixel_format DF>
struct pixel_unpack_buffer_rt<CF, DF>::impl
{
    impl() : comp_program(nullptr) {}

    std::unique_ptr<gl::depth_compositing_program>  comp_program;

    cuda::graphics_resource                         color_resource;
    cuda::graphics_resource                         depth_resource;

    gl::buffer                                      color_buffer;
    gl::buffer                                      depth_buffer;

    gl::texture                                     color_texture;
    gl::texture                                     depth_texture;
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

template <pixel_format CF, pixel_format DF>
pixel_unpack_buffer_rt<CF, DF>::~pixel_unpack_buffer_rt()
{
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::color_type* pixel_unpack_buffer_rt<CF, DF>::color()
{
    return static_cast<color_type*>(impl_->color_resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::depth_type* pixel_unpack_buffer_rt<CF, DF>::depth()
{
    return static_cast<depth_type*>(impl_->depth_resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::color_type const* pixel_unpack_buffer_rt<CF, DF>::color() const
{
    return static_cast<color_type const*>(impl_->color_resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::depth_type const* pixel_unpack_buffer_rt<CF, DF>::depth() const
{
    return static_cast<depth_type const*>(impl_->depth_resource.dev_ptr());
}

template <pixel_format CF, pixel_format DF>
typename pixel_unpack_buffer_rt<CF, DF>::ref_type pixel_unpack_buffer_rt<CF, DF>::ref()
{
    return ref_type( color(), depth() );
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::begin_frame()
{
    if (impl_->color_resource.map() == 0)
    {
        throw std::runtime_error("bad color resource mapped");
    }

    if (depth_traits::format != PF_UNSPECIFIED && impl_->depth_resource.map() == 0)
    {
        throw std::runtime_error("bad depth resource mapped");
    }
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::end_frame()
{
    pixel_format_info cinfo = map_pixel_format(color_traits::format);

    impl_->color_resource.unmap();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->color_buffer.get());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), cinfo.format, cinfo.type, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format);

        impl_->depth_resource.unmap();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->depth_buffer.get());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), dinfo.format, dinfo.type, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::resize(size_t w, size_t h)
{
    pixel_format_info cinfo = map_pixel_format(color_traits::format);

    // gl texture
    impl_->color_texture.reset( gl::create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, cinfo.internal_format, w, h, 0, cinfo.format, cinfo.type, 0);

    // gl buffer
    impl_->color_buffer.reset( gl::create_buffer() );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->color_buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * cinfo.size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register buffer object with CUDA
    impl_->color_resource.register_buffer(impl_->color_buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard);

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format);

        // gl texture
        impl_->depth_texture.reset( gl::create_texture() );

        glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, dinfo.internal_format, w, h, 0, dinfo.format, dinfo.type, 0);

        // gl buffer
        impl_->depth_buffer.reset( gl::create_buffer() );

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->depth_buffer.get());
        glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * dinfo.size, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // register buffer object with CUDA
        impl_->depth_resource.register_buffer(impl_->depth_buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard);

        if (!impl_->comp_program)
        {
            impl_->comp_program.reset(new gl::depth_compositing_program);
        }
    }

    render_target::resize(w, h);
}

template <pixel_format CF, pixel_format DF>
void pixel_unpack_buffer_rt<CF, DF>::display_color_buffer() const
{
    if (depth_traits::format == PF_UNSPECIFIED)
    {
        gl::blend_texture(impl_->color_texture.get());
    }
    else
    {
        glPushAttrib( GL_TEXTURE_BIT | GL_ENABLE_BIT );

        glEnable(GL_DEPTH_TEST);

        impl_->comp_program->enable();
        impl_->comp_program->set_textures( impl_->color_texture.get(), impl_->depth_texture.get());

        gl::draw_full_screen_quad();

        impl_->comp_program->disable();

        glPopAttrib();
    }
}

} // visionaray
