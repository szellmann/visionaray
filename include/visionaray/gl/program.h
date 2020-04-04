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
#else
#include "types.h"
#endif

#include <visionaray/detail/macros.h>
#include <visionaray/detail/platform.h>

namespace visionaray
{
namespace gl
{

class shader;

class program
{
public:

    explicit program(GLuint name = 0);

   ~program();

    void destroy();

    void reset(GLuint name = 0);

    GLuint get() const;

    void attach_shader(shader const& s) const;
    void detach_shader(shader const& s) const;

    void link() const;
    void enable() const;
    void disable() const;
    bool check_attached(shader const& s) const;
    bool check_linked() const;

protected:

    VSNRAY_NOT_COPYABLE(program)

    // This program
    GLuint name_;

    // The previously bound program
    GLuint old_;

};

} // gl
} // visionaray

#endif // VSNRAY_GL_PROGRAM_H
