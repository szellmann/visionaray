// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstring>
#ifndef NDEBUG
#include <iostream>
#endif

#include <GL/glew.h>

#include <visionaray/gl/handle.h>
#include <visionaray/gl/util.h>

#include "aligned_vector.h"
#include "make_unique.h"

#define VSNRAY_CPU_BUFFER_TEX 1

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Depth compositing frag prog
//

class depth_compositing_program
{
public:

    depth_compositing_program()
        : program_(glCreateProgram())
        , frag_(glCreateShader(GL_FRAGMENT_SHADER))
    {
        auto source =
            "uniform sampler2D color_tex;                                                       \n"
            "uniform sampler2D depth_tex;                                                       \n"
            "                                                                                   \n"
            "void main(void)                                                                    \n"
            "{                                                                                  \n"
            "    gl_FragColor = texture2D(color_tex, gl_TexCoord[0].xy);                        \n"
            "    gl_FragDepth = texture2D(depth_tex, gl_TexCoord[0].xy).x;                      \n"
            "}                                                                                  \n"
            ;

        GLint len = static_cast<GLint>(std::strlen(source));

        glShaderSource(frag_, 1, &source, &len);

        glCompileShader(frag_);
        if (!check_shader_compiled())
        {
            return;
        }

        program_ = glCreateProgram();
        glAttachShader(program_, frag_);

        glLinkProgram(program_);
        if (!check_program_linked())
        {
            return;
        }

        color_loc_ = glGetUniformLocation(program_, "color_tex");
        depth_loc_ = glGetUniformLocation(program_, "depth_tex");
    }

   ~depth_compositing_program()
    {
        glDetachShader(program_, frag_);
        glDeleteShader(frag_);
        glDeleteProgram(program_);
    }

    void enable() const
    {
        glUseProgram(program_);
    }

    void disable() const
    {
        glUseProgram(0);
    }

    void set_textures(GLuint color, GLuint depth) const
    {
        glUniform1i(color_loc_, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, color);

        glUniform1i(depth_loc_, 1);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, depth);
    }

private:

    // The program
    GLuint program_;

    // The fragment shader
    GLuint frag_;

    // Uniform location of color texture
    GLint color_loc_;

    // Uniform location of depth texture
    GLint depth_loc_;


    bool check_shader_compiled() const
    {
        GLint success = GL_FALSE;
        glGetShaderiv(frag_, GL_COMPILE_STATUS, &success);

        if (!success)
        {
            GLint length = 0;
            glGetShaderiv(frag_, GL_INFO_LOG_LENGTH, &length);

            if (length > 0)
            {
                std::vector<GLchar> buf( static_cast<size_t>(length) );
                glGetShaderInfoLog( frag_, length, &length, buf.data() );
                std::cerr << std::string(buf.data()) << std::endl;
            }

            return false;
        }

        return true;
    }

    bool check_program_linked() const
    {
        GLint success = GL_FALSE;
        glGetProgramiv(program_, GL_LINK_STATUS, &success);

        if (!success)
        {
            GLint length = 0;
            glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);

            if (length > 0)
            {
                std::vector<GLchar> buf( static_cast<size_t>(length) );
                glGetProgramInfoLog( program_, length, &length, buf.data() );
                std::cerr << std::string(buf.data()) << std::endl;
            }

            return false;
        }

        return true;
    }

};


//-------------------------------------------------------------------------------------------------
// Helper functions
//

void alloc_texture(pixel_format_info info, size_t w, size_t h)
{
    if (glTexStorage2D)
    {
        glTexStorage2D(GL_TEXTURE_2D, 1, info.internal_format, w, h);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, info.internal_format, w, h, 0, info.format, info.type, 0);
    }
}

} // detail


//-------------------------------------------------------------------------------------------------
// Private implementation
//

template <pixel_format CF, pixel_format DF>
struct cpu_buffer_rt<CF, DF>::impl
{
    impl() : comp_program(nullptr) {}

    std::unique_ptr<detail::depth_compositing_program>  comp_program;
    aligned_vector<color_type>                          color_buffer;
    aligned_vector<depth_type>                          depth_buffer;

    gl::texture                                         color_texture;
    gl::texture                                         depth_texture;
};


//-------------------------------------------------------------------------------------------------
// cpu_buffer_rt
//

template <pixel_format CF, pixel_format DF>
cpu_buffer_rt<CF, DF>::cpu_buffer_rt()
    : impl_(new impl)
{
}

template <pixel_format CF, pixel_format DF>
cpu_buffer_rt<CF, DF>::~cpu_buffer_rt()
{
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::color_type* cpu_buffer_rt<CF, DF>::color()
{
    return impl_->color_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::depth_type* cpu_buffer_rt<CF, DF>::depth()
{
    return impl_->depth_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::color_type const* cpu_buffer_rt<CF, DF>::color() const
{
    return impl_->color_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::depth_type const* cpu_buffer_rt<CF, DF>::depth() const
{
    return impl_->depth_buffer.data();
}

template <pixel_format CF, pixel_format DF>
typename cpu_buffer_rt<CF, DF>::ref_type cpu_buffer_rt<CF, DF>::ref()
{
    return typename cpu_buffer_rt<CF, DF>::ref_type( color(), depth() );
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::begin_frame()
{
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::end_frame()
{
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::resize(size_t w, size_t h)
{
    render_target::resize(w, h);

    // Allocate storage

    pixel_format_info cinfo = map_pixel_format(color_traits::format);
    impl_->color_buffer.resize( w * h * cinfo.size );

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format);
        impl_->depth_buffer.resize( w * h * dinfo.size );
    }

#if VSNRAY_CPU_BUFFER_TEX

    glPushAttrib( GL_TEXTURE_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_ENABLE_BIT );

    // Allocate texture storage

    impl_->color_texture.reset( gl::create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    detail::alloc_texture(cinfo, w, h);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        pixel_format_info dinfo = map_pixel_format(depth_traits::format);

        impl_->depth_texture.reset( gl::create_texture() );

        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

        detail::alloc_texture(dinfo, w, h);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        if (!impl_->comp_program)
        {
            impl_->comp_program.reset(new detail::depth_compositing_program);
        }
    }

    glPopAttrib();

#endif
}

template <pixel_format CF, pixel_format DF>
void cpu_buffer_rt<CF, DF>::display_color_buffer() const
{
#if VSNRAY_CPU_BUFFER_TEX

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        glPushAttrib( GL_TEXTURE_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_ENABLE_BIT );

        // Update color texture

        pixel_format_info cinfo = map_pixel_format(color_traits::format);

        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

        glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                width(),
                height(),
                cinfo.format,
                cinfo.type,
                impl_->color_buffer.data()
                );


        // Update depth texture

        pixel_format_info dinfo = map_pixel_format(depth_traits::format);

        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

        glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                width(),
                height(),
                dinfo.format,
                dinfo.type,
                impl_->depth_buffer.data()
                );


        // Combine textures using a shader

        glEnable(GL_DEPTH_TEST);

        impl_->comp_program->enable();
        impl_->comp_program->set_textures( impl_->color_texture.get(), impl_->depth_texture.get());

        gl::draw_full_screen_quad();

        impl_->comp_program->disable();

        glPopAttrib();
    }
    else
    {
        pixel_format_info cinfo = map_pixel_format(color_traits::format);

        glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

        glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                width(),
                height(),
                cinfo.format,
                cinfo.type,
                impl_->color_buffer.data()
                );

        gl::blend_texture(impl_->color_texture.get());
    }

#else

    // Use glDrawPixels

    if (depth_traits::format != PF_UNSPECIFIED)
    {
        glPushAttrib( GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_ENABLE_BIT );

        glEnable(GL_STENCIL_TEST);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glStencilFunc(GL_ALWAYS, 1, 1);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

        pixel_format_info dinfo = map_pixel_format(depth_traits::format);
        gl::blend_pixels( width(), height(), dinfo.format, dinfo.type, impl_->depth_buffer.data() );

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilFunc(GL_EQUAL, 1, 1);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glDisable(GL_DEPTH_TEST);

        pixel_format_info cinfo = map_pixel_format(color_traits::format);
        gl::blend_pixels( width(), height(), cinfo.format, cinfo.type, impl_->color_buffer.data() );

        glPopAttrib();
    }
    else
    {
        pixel_format_info info = map_pixel_format(color_traits::format);
        gl::blend_pixels( width(), height(), info.format, info.type, impl_->color_buffer.data() );
    }

#endif
}

} // visionaray
