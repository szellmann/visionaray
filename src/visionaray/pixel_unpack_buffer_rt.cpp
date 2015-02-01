// This file is distributed under the MIT license.
// See the LICENSE file for details.

#if defined(VSNRAY_HAVE_OPENGL) && defined(VSNRAY_HAVE_CUDA)

#include <stdexcept>

#include <GL/glew.h>

#include <cuda_gl_interop.h>

#include <visionaray/pixel_unpack_buffer_rt.h>

#include "cuda/graphics_resource.h"
#include "gl/handle.h"
#include "gl/util.h"


namespace visionaray
{

struct pixel_unpack_buffer_rt::impl
{
    impl(pixel_format cf, pixel_format df)
        : color_format(cf)
        , depth_format(df)
    {
    }

    cuda::graphics_resource resource;
    gl::buffer              buffer;
    gl::texture             texture;
    pixel_format            color_format;
    pixel_format            depth_format;
};

pixel_unpack_buffer_rt::pixel_unpack_buffer_rt(pixel_format cf, pixel_format df)
    : impl_(new impl(cf, df))
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

pixel_unpack_buffer_rt::~pixel_unpack_buffer_rt()
{
}

void* pixel_unpack_buffer_rt::color()
{
    return impl_->resource.dev_ptr();
}

void* pixel_unpack_buffer_rt::depth()
{
    throw std::runtime_error("not implemented yet");
}

void const* pixel_unpack_buffer_rt::color() const
{
    return impl_->resource.dev_ptr();
}

void const* pixel_unpack_buffer_rt::depth() const
{
    throw std::runtime_error("not implemented yet");
}

void pixel_unpack_buffer_rt::begin_frame_impl()
{
    if (impl_->resource.map() == 0)
    {
        throw std::runtime_error("bad resource mapped");
    }
}

void pixel_unpack_buffer_rt::end_frame_impl()
{
    pixel_format_info cinfo = map_pixel_format(impl_->color_format);

    impl_->resource.unmap();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, impl_->texture.get());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->buffer.get());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), cinfo.format, cinfo.type, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void pixel_unpack_buffer_rt::resize_impl(size_t w, size_t h)
{
    pixel_format_info cinfo = map_pixel_format(impl_->color_format);

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
}

void pixel_unpack_buffer_rt::display_color_buffer_impl() const
{
    gl::blend_texture(impl_->texture.get());
}

} // visionaray

#endif // VSNRAY_HAVE_OPENGL && VSNRAY_HAVE_CUDA


