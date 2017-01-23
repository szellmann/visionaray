// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/gl/handle.h>

namespace visionaray
{
namespace gl
{

void buffer::destroy()
{
    glDeleteBuffers(1, &name_);
}

void framebuffer::destroy()
{
    glDeleteFramebuffers(1, &name_);
}

void renderbuffer::destroy()
{
    glDeleteRenderbuffers(1, &name_);
}

void texture::destroy()
{
    glDeleteTextures(1, &name_);
}

void vertex_array::destroy()
{
    glDeleteVertexArrays(1, &name_);
}


GLuint create_buffer()
{
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    return buf;
}

GLuint create_framebuffer()
{
    GLuint buf = 0;
    glGenFramebuffers(1, &buf);
    return buf;
}

GLuint create_renderbuffer()
{
    GLuint buf = 0;
    glGenRenderbuffers(1, &buf);
    return buf;
}

GLuint create_texture()
{
    GLuint buf = 0;
    glGenTextures(1, &buf);
    return buf;
}

GLuint create_vertex_array()
{
    GLuint buf = 0;
    glGenVertexArrays(1, &buf);
    return buf;
}

} // gl
} // visionaray
