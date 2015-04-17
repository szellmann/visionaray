// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdio>

#include <exception>
#include <iostream>
#include <ostream>

#include <visionaray/detail/platform.h>

#ifdef VSNRAY_OS_WIN32
#include <windows.h>
#endif

#include <GL/glew.h>

#include <visionaray/gl/util.h>
#include <visionaray/math/math.h>

#include "../util.h"


namespace gl = visionaray::gl;


#if defined(GL_KHR_debug)


static char const* get_debug_type_string(GLenum type)
{
    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:
        return "error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "deprecated behavior detected";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "undefined behavior detected";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "portablility warning";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "performance warning";
    case GL_DEBUG_TYPE_OTHER:
        return "other";
    case GL_DEBUG_TYPE_MARKER:
        return "marker";
    }

    return "{unknown type}";
}

static void debug_callback(
        GLenum          /*source*/,
        GLenum          type,
        GLuint          /*id*/,
        GLenum          /*severity*/,
        GLsizei         /*length*/,
        const GLchar*   message,
        GLvoid*         /*userParam*/
        )
{
    std::cerr << "GL " << get_debug_type_string(type) << " " << message << std::endl;
    std::cerr << visionaray::util::backtrace() << std::endl;
    throw std::exception();

    if (type == GL_DEBUG_TYPE_ERROR)
    {
#ifdef _WIN32
        if (IsDebuggerPresent())
            DebugBreak();
#endif
    }
}


#endif // GL_KHR_debug


namespace visionaray
{

void gl::init_debug_callback()
{
#if defined(GL_KHR_debug)
    if (GLEW_KHR_debug)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback((GLDEBUGPROC)debug_callback, 0);
    }
#elif defined(GL_ARB_debug_output)
    if (GLEW_ARB_debug_output)
    {
    }
#endif
}

std::string gl::last_error()
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        return std::string(reinterpret_cast<char const*>(glewGetErrorString(err)));
    }
    return "";
}

void gl::draw_full_screen_quad()
{
    glPushAttrib(GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glActiveTexture(GL_TEXTURE0);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

void gl::blend_texture(GLuint texture, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);

    draw_full_screen_quad();

    glPopAttrib();
}

void gl::blend_pixels(GLsizei w, GLsizei h, GLenum format, GLenum type, GLvoid const* pixels, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    recti vp = gl::viewport();
    glWindowPos2i(vp[0], vp[1]);

    GLfloat scalex = vp[2] / static_cast<GLfloat>(w);
    GLfloat scaley = vp[3] / static_cast<GLfloat>(h);

    glPixelZoom(scalex, scaley);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glDrawPixels(w, h, format, type, pixels);

    glPopAttrib();
}

recti gl::viewport()
{
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, &vp[0]);
    return recti(vp[0], vp[1], vp[2], vp[3]);
}

} // visionaray
