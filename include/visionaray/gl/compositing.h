// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_COMPOSITING_H
#define VSNRAY_GL_COMPOSITING_H 1

#include <iostream>
#include <ostream>
#include <vector>

#include <GL/glew.h>

#include <visionaray/pixel_format.h>

#include "handle.h"
#include "util.h"

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// GLSL-based depth compositor
//

class depth_compositor
{
public:

    depth_compositor()
        : program_(glCreateProgram())
        , frag_(glCreateShader(GL_FRAGMENT_SHADER))
    {
        auto source = R"(
            uniform sampler2D color_tex;
            uniform sampler2D depth_tex;

            void main(void)
            {
                gl_FragColor = texture2D(color_tex, gl_TexCoord[0].xy);
                gl_FragDepth = texture2D(depth_tex, gl_TexCoord[0].xy).x;
            }
            )";

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

   ~depth_compositor()
    {
        glDetachShader(program_, frag_);
        glDeleteShader(frag_);
        glDeleteProgram(program_);
    }

    void composite_textures() const
    {
        glEnable(GL_DEPTH_TEST);

        enable_program();

        gl::draw_full_screen_quad();

        disable_program();
    }

    void display_color_texture() const
    {
        gl::blend_texture(color_texture_.get());
    }

    void setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h)
    {
        color_texture_.reset( create_texture() );

        glBindTexture(GL_TEXTURE_2D, color_texture_.get());

        set_texture_params();

        alloc_texture(info, w, h);
    }

    void setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h)
    {
        depth_texture_.reset( create_texture() );

        glBindTexture(GL_TEXTURE_2D, depth_texture_.get());

        set_texture_params();

        alloc_texture(info, w, h);
    }

    void update_color_texture(
            pixel_format_info   info,
            GLsizei             w,
            GLsizei             h,
            GLvoid const*       data
            ) const
    {
        glBindTexture(GL_TEXTURE_2D, color_texture_.get());

        gl::update_texture( info, w, h, data );
    }

    void update_depth_texture(
            pixel_format_info   info,
            GLsizei             w,
            GLsizei             h,
            GLvoid const*       data
            ) const
    {
        glBindTexture(GL_TEXTURE_2D, depth_texture_.get());

        gl::update_texture( info, w, h, data );
    }

private:

    // GL color texture handle
    gl::texture color_texture_;

    // GL color texture handle
    gl::texture depth_texture_;

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

    void enable_program() const
    {
        glUseProgram(program_);

        glUniform1i(color_loc_, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, color_texture_.get());

        glUniform1i(depth_loc_, 1);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, depth_texture_.get());
    }

    void disable_program() const
    {
        glUseProgram(0);
    }

    void set_texture_params() const
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    }

};


} // gl
} // visionaray

#endif // VSNRAY_GL_COMPOSITING_H
