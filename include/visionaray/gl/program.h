// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_PROGRAM_H
#define VSNRAY_GL_PROGRAM_H 1

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <visionaray/detail/macros.h>
#include <visionaray/detail/platform.h>

namespace visionaray
{
namespace gl
{

class program
{
public:

    explicit program(GLuint name = 0) : name_(name) {}
   ~program() { reset(); }

    void destroy()
    {
        glDeleteProgram(name_);
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

    void link() const;
    void enable() const;
    void disable() const;
    bool check_linked() const;

protected:

    VSNRAY_NOT_COPYABLE(program)

    GLuint name_;

};

} // gl
} // visionaray

#endif // VSNRAY_GL_PROGRAM_H
