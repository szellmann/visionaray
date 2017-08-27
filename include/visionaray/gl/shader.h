// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_SHADER_H
#define VSNRAY_GL_SHADER_H 1

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <visionaray/detail/macros.h>

namespace visionaray
{
namespace gl
{

class shader
{
public:

    explicit shader(GLuint name = 0) : name_(name) {}
   ~shader() { reset(); }

    void destroy()
    {
        glDeleteShader(name_);
    }

    void reset(GLuint name = 0)
    {
        if (name_ != 0)
        {
            destroy();
        }
        name_ = name;
    }

    GLuint get() const { return name_; }

    void set_source(char const* source) const;
    void compile() const;
    bool check_compiled() const;

private:

    VSNRAY_NOT_COPYABLE(shader)

    GLuint name_;
};

} // gl
} // visionaray

#endif // VSNRAY_GL_SHADER_H
