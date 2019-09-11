// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_HANDLE_H
#define VSNRAY_GL_HANDLE_H 1

#include <visionaray/config.h>

#if defined(VSNRAY_HAVE_GLEW)
#include <GL/glew.h>
#elif defined(VSNRAY_HAVE_OPENGLES)
#include <GLES2/gl2.h>
#endif

#include <visionaray/detail/macros.h>
#include <visionaray/detail/platform.h>
#include <visionaray/export.h>

namespace visionaray
{
namespace gl
{

template <typename Derived>
class handle
{
public:

    explicit handle(GLuint name = 0) : name_(name) {}
   ~handle() { reset(); }

    void reset(GLuint name = 0)
    {
        if (name_ != 0)
        {
            static_cast<Derived*>(this)->destroy();
        }
        name_ = name;
    }

    GLuint get() const { return name_; }

protected:

    VSNRAY_NOT_COPYABLE(handle)

    GLuint name_;

};


class buffer : public handle<buffer>
{
public:

    explicit buffer(GLuint name = 0) : handle<buffer>(name) {}

    VSNRAY_EXPORT void destroy();

private:

    VSNRAY_NOT_COPYABLE(buffer)

};


class framebuffer : public handle<framebuffer>
{
public:

    explicit framebuffer(GLuint name = 0) : handle<framebuffer>(name) {}

    VSNRAY_EXPORT void destroy();

private:

    VSNRAY_NOT_COPYABLE(framebuffer)

};


class renderbuffer : public handle<renderbuffer>
{
public:

    explicit renderbuffer(GLuint name = 0) : handle<renderbuffer>(name) {}

    VSNRAY_EXPORT void destroy();

private:

    VSNRAY_NOT_COPYABLE(renderbuffer)

};


class texture : public handle<texture>
{
public:

    explicit texture(GLuint name = 0) : handle<texture>(name) {}

    VSNRAY_EXPORT void destroy();

private:

    VSNRAY_NOT_COPYABLE(texture)

};


class vertex_array : public handle<vertex_array>
{
public:

    explicit vertex_array(GLuint name = 0) : handle<vertex_array>(name) {}

    VSNRAY_EXPORT void destroy();

private:

    VSNRAY_NOT_COPYABLE(vertex_array)

};


//-------------------------------------------------------------------------------------------------
// factory functions
//

VSNRAY_EXPORT GLuint create_buffer();
VSNRAY_EXPORT GLuint create_framebuffer();
VSNRAY_EXPORT GLuint create_renderbuffer();
VSNRAY_EXPORT GLuint create_texture();
VSNRAY_EXPORT GLuint create_vertex_array();

} // gl
} // visionaray

#endif // VSNRAY_GL_HANDLE_H
