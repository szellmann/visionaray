// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/gl/handle.h>

#if !(VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES)
#define glGenBuffers(X,Y)
#define glGenFramebuffers(X,Y)
#define glGenRenderbuffers(X,Y)
#define glGenTextures(X,Y)
#define glDeleteBuffers(X,Y)
#define glDeleteFramebuffers(X,Y)
#define glDeleteRenderbuffers(X,Y)
#define glDeleteTextures(X,Y)
#endif

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
#if defined(GL_VERSION_3_0) && GL_VERSION_3_0 || defined(GL_ES_VERSION_3_0) && GL_ES_VERSION_3_0
    glDeleteVertexArrays(1, &name_);
#endif
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
#if defined(GL_VERSION_3_0) && GL_VERSION_3_0 || defined(GL_ES_VERSION_3_0) && GL_ES_VERSION_3_0
    glGenVertexArrays(1, &buf);
#endif
    return buf;
}

} // gl
} // visionaray
