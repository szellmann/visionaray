// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <visionaray/cpu_buffer_rt.h>

#ifdef __CUDACC__
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
    mode current_mode;

    // If true, use PBO, otherwise copy over host
    bool direct_rendering;

    // Host render target
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> host_rt;

#ifdef __CUDACC__
    // Device render target, uses PBO
    pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> direct_rt;

    // Device render target, copy pixels over host
    gpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> indirect_rt;
#endif
};


//-------------------------------------------------------------------------------------------------
// host_device_rt
//

host_device_rt::host_device_rt(mode m, bool direct_rendering)
    : impl_(new impl)
{
    impl_->current_mode = m;
    impl_->direct_rendering = direct_rendering;
}

host_device_rt::~host_device_rt()
{
}

host_device_rt::mode& host_device_rt::current_mode()
{
    return impl_->current_mode;
}

host_device_rt::mode const& host_device_rt::current_mode() const
{
    return impl_->current_mode;
}

bool& host_device_rt::direct_rendering()
{
    return impl_->direct_rendering;
}

bool const& host_device_rt::direct_rendering() const
{
    return impl_->direct_rendering;
}

host_device_rt::color_type const* host_device_rt::color() const
{
    return impl_->host_rt.color();
}

host_device_rt::ref_type host_device_rt::ref()
{
    if (impl_->current_mode == CPU)
    {
        return impl_->host_rt.ref();
    }
    else
    {
#ifdef __CUDACC__
        if (impl_->direct_rendering)
        {
            return impl_->direct_rt.ref();
        }
        else
        {
            return impl_->indirect_rt.ref();
        }
#else
        assert(0);
        return {};
#endif
    }
}

void host_device_rt::clear_color_buffer(vec4 const& color)
{
    impl_->host_rt.clear_color_buffer(color);
#ifdef __CUDACC__
    if (impl_->direct_rendering)
    {
        impl_->direct_rt.clear_color_buffer(color);
    }
    else
    {
        impl_->indirect_rt.clear_color_buffer(color);
    }
#endif
}

void host_device_rt::begin_frame()
{
    if (impl_->current_mode == CPU)
    {
        impl_->host_rt.begin_frame();
    }
#ifdef __CUDACC__
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt.begin_frame();
        }
        else
        {
            impl_->indirect_rt.begin_frame();
        }
    }
#endif
}

void host_device_rt::end_frame()
{
    if (impl_->current_mode == CPU)
    {
        impl_->host_rt.end_frame();
    }
#ifdef __CUDACC__
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt.end_frame();
        }
        else
        {
            impl_->indirect_rt.end_frame();
        }
    }
#endif
}

void host_device_rt::resize(int w, int h)
{
    render_target::resize(w, h);

    impl_->host_rt.resize(w, h);
#ifdef __CUDACC__
    if (impl_->direct_rendering)
    {
        impl_->direct_rt.resize(w, h);
    }
    else
    {
        impl_->indirect_rt.resize(w, h);
    }
#endif
}

void host_device_rt::display_color_buffer() const
{
    if (impl_->current_mode == CPU)
    {
        impl_->host_rt.display_color_buffer();
    }
#ifdef __CUDACC__
    else
    {
        if (impl_->direct_rendering)
        {
            impl_->direct_rt.display_color_buffer();
        }
        else
        {
            impl_->indirect_rt.display_color_buffer();
        }
    }
#endif
}

} // visionaray
