// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <visionaray/detail/macros.h>

#include "program.h"
#include "shader.h"

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
    glDeleteProgram(name_);
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
    glAttachShader(name_, s.get());
}

void program::detach_shader(shader const& s) const
{
    glDetachShader(name_, s.get());
}

void program::link() const
{
    glLinkProgram(name_);
}

void program::enable() const
{
    glGetIntegerv(GL_CURRENT_PROGRAM, const_cast<GLint*>(reinterpret_cast<GLint const*>(&old_)));
    glUseProgram(name_);
}

void program::disable() const
{
    glUseProgram(old_);
}

bool program::check_attached(shader const& s) const
{
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
}

bool program::check_linked() const
{
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
}

} // gl
} // visionaray
