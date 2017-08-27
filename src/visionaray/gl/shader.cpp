// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <visionaray/gl/shader.h>

namespace visionaray
{
namespace gl
{

void shader::set_source(char const* source) const
{
  GLint len = static_cast<GLint>(std::strlen(source));

  glShaderSource(name_, 1, &source, &len);
}

void shader::compile() const
{
    glCompileShader(name_);
}

bool shader::check_compiled() const
{
    GLint success = GL_FALSE;
    glGetShaderiv(name_, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        GLint length = 0;
        glGetShaderiv(name_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            std::vector<GLchar> buf(static_cast<size_t>(length));
            glGetShaderInfoLog(name_, length, &length, buf.data());
            std::cerr << std::string(buf.data()) << '\n';
        }

        return false;
    }

    return true;
}

} // gl
} // visionaray
