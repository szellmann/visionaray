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

    explicit shader(GLuint name = 0);
   ~shader();

    void destroy();

    void reset(GLuint name = 0);

    GLuint get() const;

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
