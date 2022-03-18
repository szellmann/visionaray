// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <cassert>
#include <utility>

#include <GL/glew.h>

#include <visionaray/cpu_buffer_rt.h>

#if VSNRAY_HAVE_CUDA
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include "host_device_rt.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct host_device_rt::impl
{
    // CPU or GPU rendering
    mode_type mode;

    // If true, render target uses double buffering
    bool double_buffering;

    // If true, use PBO, otherwise copy over host
    bool direct_rendering;

    // Framebuffer color space, either RGB or SRGB
    color_space_type color_space;

    // Host render target
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> host_rt[2];

#if VSNRAY_HAVE_CUDA
    // Device render target, uses PBO
    pixel_unpack_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> direct_rt[2];

    // Device render target, copy pixels over host
    gpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> indirect_rt[2];
#endif

    // Index of front and back buffer, either 0 or 1
    bool buffer_index[2];
};


//-------------------------------------------------------------------------------------------------
// host_device_rt
//

host_device_rt::host_device_rt(
        mode_type mode,
        bool double_buffering,
        bool direct_rendering,
        color_space_type color_space
        )
    : impl_(new impl)
{
    impl_->mode = mode;
    set_double_buffering(double_buffering);
    impl_->direct_rendering = direct_rendering;
    impl_->color_space = color_space;

}

host_device_rt::~host_device_rt()
{
}

host_device_rt::mode_type& host_device_rt::mode()
{
    return impl_->mode;
}

host_device_rt::mode_type const& host_device_rt::mode() const
{
    return impl_->mode;
}

void host_device_rt::set_double_buffering(bool double_buffering)
{
    impl_->double_buffering = double_buffering;

    if (impl_->double_buffering)
    {
        impl_->buffer_index[0] = 0;
        impl_->buffer_index[1] = 1;
    }
    else
    {
        // Both indices the same, so when we swap buffers, in the
        // single buffering case nothing will ever get swapped
        impl_->buffer_index[0] = 0;
        impl_->buffer_index[1] = 0;
    }
}

bool host_device_rt::get_double_buffering() const
{
    return impl_->double_buffering;
}

bool& host_device_rt::direct_rendering()
{
    return impl_->direct_rendering;
}

bool const& host_device_rt::direct_rendering() const
{
    return impl_->direct_rendering;
}

host_device_rt::color_space_type& host_device_rt::color_space()
{
    return impl_->color_space;
}

host_device_rt::color_space_type const& host_device_rt::color_space() const
{
    return impl_->color_space;
}

void host_device_rt::swap_buffers()
{
    std::swap(impl_->buffer_index[Front], impl_->buffer_index[Back]);
}

host_device_rt::color_type const* host_device_rt::color(buffer buf) const
{
    return impl_->host_rt[impl_->buffer_index[buf]].color();
}

host_device_rt::ref_type host_device_rt::ref(buffer buf)
{
    if (impl_->mode == CPU)
    {
        return impl_->host_rt[impl_->buffer_index[buf]].ref();
    }
    else
    {
#if VSNRAY_HAVE_CUDA
        if (impl_->direct_rendering)
        {
            return impl_->direct_rt[impl_->buffer_index[buf]].ref();
        }
        else
        {
            return impl_->indirect_rt[impl_->buffer_index[buf]].ref();
        }
#else
        assert(0);
        return {};
#endif
    }
}

void host_device_rt::clear(vec4 const& color, buffer buf)
{
    impl_->host_rt[impl_->buffer_index[buf]].clear_color_buffer(color);
    impl_->host_rt[impl_->buffer_index[buf]].clear_accum_buffer(color);
#if VSNRAY_HAVE_CUDA
    if (impl_->direct_rendering)
    {
        impl_->direct_rt[impl_->buffer_index[buf]].clear_color_buffer(color);
    }
    else
    {
        impl_->indirect_rt[impl_->buffer_index[buf]].clear_color_buffer(color);
    }
#endif
}

void host_device_rt::begin_frame(buffer buf)
{
    if (impl_->mode == CPU)
    {
        impl_->host_rt[impl_->buffer_index[buf]].begin_frame();
    }
#if VSNRAY_HAVE_CUDA
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt[impl_->buffer_index[buf]].begin_frame();
        }
        else
        {
            impl_->indirect_rt[impl_->buffer_index[buf]].begin_frame();
        }
    }
#endif
}

void host_device_rt::end_frame(buffer buf)
{
    if (impl_->mode == CPU)
    {
        impl_->host_rt[impl_->buffer_index[buf]].end_frame();
    }
#if VSNRAY_HAVE_CUDA
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt[impl_->buffer_index[buf]].end_frame();
        }
        else
        {
            impl_->indirect_rt[impl_->buffer_index[buf]].end_frame();
        }
    }
#endif
}

void host_device_rt::resize(int w, int h)
{
    render_target::resize(w, h);

    int num_buffers = impl_->double_buffering ? 2 : 1;
    for (int buf = 0; buf < num_buffers; ++buf)
    {
        impl_->host_rt[buf].resize(w, h);
#if VSNRAY_HAVE_CUDA
        if (impl_->direct_rendering)
        {
            impl_->direct_rt[buf].resize(w, h);
        }
        else
        {
            impl_->indirect_rt[buf].resize(w, h);
        }
#endif
    }
}

void host_device_rt::display_color_buffer(buffer buf) const
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Store OpenGL state
    GLboolean prev_srgb_enabled = glIsEnabled(GL_FRAMEBUFFER_SRGB);

    if (impl_->color_space == SRGB)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }


    if (impl_->mode == CPU)
    {
        impl_->host_rt[impl_->buffer_index[buf]].display_color_buffer();
    }
#if VSNRAY_HAVE_CUDA
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt[impl_->buffer_index[buf]].display_color_buffer();
        }
        else
        {
            impl_->indirect_rt[impl_->buffer_index[buf]].display_color_buffer();
        }
    }
#endif

    if (prev_srgb_enabled)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }
}

} // visionaray
