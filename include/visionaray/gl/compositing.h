// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_COMPOSITING_H
#define VSNRAY_GL_COMPOSITING_H

#include <GL/glew.h>

namespace visionaray
{
namespace gl
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


} // gl
} // visionaray

#endif // VSNRAY_GL_COMPOSITING_H
