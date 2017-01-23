// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <visionaray/gl/program.h>

namespace visionaray
{
namespace gl
{

void program::link() const
{
    glLinkProgram(name_);
}

void program::enable() const
{
    glUseProgram(name_);
}

void program::disable() const
{
    glUseProgram(0);
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
