// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <visionaray/gl/program.h>
#include <visionaray/gl/shader.h>
#include <visionaray/detail/macros.h>

namespace visionaray
{
namespace gl
{

program::program(GLuint name)
    : name_(name)
    , old_(0)
{
}

program::~program()
{
    reset();
}

void program::destroy()
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glDeleteProgram(name_);
#endif
}

void program::reset(GLuint name)
{
    if (name_ != 0)
    {
        destroy();
    }

    name_ = name;
}

GLuint program::get() const
{
    return name_;
}

void program::attach_shader(shader const& s) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glAttachShader(name_, s.get());
#else
    VSNRAY_UNUSED(s);
#endif
}

void program::detach_shader(shader const& s) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glDetachShader(name_, s.get());
#else
    VSNRAY_UNUSED(s);
#endif
}

void program::link() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glLinkProgram(name_);
#endif
}

void program::enable() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glGetIntegerv(GL_CURRENT_PROGRAM, const_cast<GLint*>(reinterpret_cast<GLint const*>(&old_)));
    glUseProgram(name_);
#endif
}

void program::disable() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    glUseProgram(old_);
#endif
}

bool program::check_attached(shader const& s) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    // Check if program was created at all.
    if (name_ == 0)
    {
        return false;
    }


    // Check list of attached shaders.
    static const GLsizei MaxCount = 16;

    GLsizei count = 0;
    GLuint shaders[MaxCount];

    glGetAttachedShaders(name_, MaxCount, &count, shaders);

    for (GLsizei i = 0; i < count; ++i)
    {
        if (shaders[i] == s.get())
        {
            return true;
        }
    }

    return false;
#else
    VSNRAY_UNUSED(s);
    return false;
#endif
}

bool program::check_linked() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    GLint success = GL_FALSE;
    glGetProgramiv(name_, GL_LINK_STATUS, &success);

    if (!success)
    {
        GLint length = 0;
        glGetProgramiv(name_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            std::vector<GLchar> buf(static_cast<size_t>(length));
            glGetProgramInfoLog(name_, length, &length, buf.data());
            std::cerr << std::string(buf.data()) << '\n';
        }

        return false;
    }

    return true;
#else
    return false;
#endif
}

} // gl
} // visionaray
