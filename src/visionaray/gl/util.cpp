// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/gl/util.h>
#include <visionaray/math/math.h>

#include "../util.h"


namespace gl = visionaray::gl;


namespace visionaray
{

std::string gl::last_error()
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
#if VSNRAY_HAVE_GLEW
        return std::string(reinterpret_cast<char const*>(glewGetErrorString(err)));
#endif
    }
    return "";
}

void gl::alloc_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
#if defined(GL_VERSION_4_2) && GL_VERSION_4_2 || defined(GL_ES_VERSION_3_0) && GL_ES_VERSION_3_0
    if (glTexStorage2D)
    {
        glTexStorage2D(GL_TEXTURE_2D, 1, info.internal_format, w, h);
    }
    else
#endif
    {
        glTexImage2D(GL_TEXTURE_2D, 0, info.internal_format, w, h, 0, info.format, info.type, 0);
    }
}

void gl::update_texture(
        pixel_format_info   info,
        GLsizei             x,
        GLsizei             y,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       pixels
        )
{
    glTexSubImage2D(
            GL_TEXTURE_2D,
            0, // TODO
            x,
            y,
            w,
            h,
            info.format,
            info.type,
            pixels
            );
}

void gl::update_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       pixels
        )
{
    glTexSubImage2D(
            GL_TEXTURE_2D,
            0, // TODO
            0,
            0,
            w,
            h,
            info.format,
            info.type,
            pixels
            );
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

#if defined(VSNRAY_OPENGL_LEGACY)
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
#endif

recti gl::viewport()
{
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, &vp[0]);
    return recti(vp[0], vp[1], vp[2], vp[3]);
}

} // visionaray
