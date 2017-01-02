// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstring>

#include <GL/glew.h>

#include <visionaray/gl/compositing.h>
#include <visionaray/aligned_vector.h>

#include "color_conversion.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Private implementation
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
struct cpu_buffer_rt<ColorFormat, DepthFormat>::impl
{
    impl() : compositor(nullptr) {}

    std::unique_ptr<gl::depth_compositor>   compositor;

    aligned_vector<color_type>              color_buffer;
    aligned_vector<depth_type>              depth_buffer;
};


//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

template <pixel_format ColorFormat, pixel_format DepthFormat>
cpu_buffer_rt<ColorFormat, DepthFormat>::cpu_buffer_rt()
    : impl_(new impl)
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
cpu_buffer_rt<ColorFormat, DepthFormat>::~cpu_buffer_rt()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::color_type* cpu_buffer_rt<ColorFormat, DepthFormat>::color()
{
    return impl_->color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::depth_type* cpu_buffer_rt<ColorFormat, DepthFormat>::depth()
{
    return impl_->depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::color_type const* cpu_buffer_rt<ColorFormat, DepthFormat>::color() const
{
    return impl_->color_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::depth_type const* cpu_buffer_rt<ColorFormat, DepthFormat>::depth() const
{
    return impl_->depth_buffer.data();
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
typename cpu_buffer_rt<ColorFormat, DepthFormat>::ref_type cpu_buffer_rt<ColorFormat, DepthFormat>::ref()
{
    return typename cpu_buffer_rt<ColorFormat, DepthFormat>::ref_type(
            color(),
            depth(),
            width(),
            height()
            );
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::clear_color_buffer(vec4 const& c)
{
    // Convert from RGBA32F to internal color format
    color_type cc;
    convert(
        pixel_format_constant<ColorFormat>{},
        pixel_format_constant<PF_RGBA32F>{},
        cc,
        c
        );

    std::fill(impl_->color_buffer.begin(), impl_->color_buffer.end(), cc);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::clear_depth_buffer(float d)
{
    // Convert from DEPTH32F to internal depth format
    depth_type dd;
    convert(
        pixel_format_constant<DepthFormat>{},
        pixel_format_constant<PF_DEPTH32F>{},
        dd,
        d
        );

    std::fill(impl_->depth_buffer.begin(), impl_->depth_buffer.end(), dd);
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::begin_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::end_frame()
{
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);


    // Allocate storage

    impl_->color_buffer.resize(w * h);

    if (DepthFormat != PF_UNSPECIFIED)
    {
        impl_->depth_buffer.resize(w * h);
    }

    if (!impl_->compositor)
    {
        impl_->compositor.reset(new gl::depth_compositor);
    }

    // Allocate texture storage

    pixel_format_info cinfo = map_pixel_format(ColorFormat);

    impl_->compositor->setup_color_texture(cinfo, w, h);


    if (DepthFormat != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        impl_->compositor->setup_depth_texture(dinfo, w, h);
    }
}

template <pixel_format ColorFormat, pixel_format DepthFormat>
void cpu_buffer_rt<ColorFormat, DepthFormat>::display_color_buffer() const
{
    if (DepthFormat != PF_UNSPECIFIED)
    {
        glPushAttrib( GL_TEXTURE_BIT );

        // Update color texture

        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        impl_->compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                impl_->color_buffer.data()
                );


        // Update depth texture

        pixel_format_info dinfo = map_pixel_format(DepthFormat);

        impl_->compositor->update_depth_texture(
                dinfo,
                width(),
                height(),
                impl_->depth_buffer.data()
                );


        // Combine textures using a shader

        impl_->compositor->composite_textures();

        glPopAttrib();
    }
    else
    {
        pixel_format_info cinfo = map_pixel_format(ColorFormat);

        impl_->compositor->update_color_texture(
                cinfo,
                width(),
                height(),
                impl_->color_buffer.data()
                );

        impl_->compositor->display_color_texture();
    }
}

} // visionaray
