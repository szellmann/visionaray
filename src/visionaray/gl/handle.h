// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_HANDLE_H
#define VSNRAY_GL_HANDLE_H

#include <visionaray/detail/macros.h>

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

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

    explicit buffer(GLuint name = 0) : handle(name) {}

    void destroy();

private:

    VSNRAY_NOT_COPYABLE(buffer)

};


class framebuffer : public handle<framebuffer>
{
public:

    explicit framebuffer(GLuint name = 0) : handle(name) {}

    void destroy();

private:

    VSNRAY_NOT_COPYABLE(framebuffer)

};


class renderbuffer : public handle<renderbuffer>
{
public:

    explicit renderbuffer(GLuint name = 0) : handle(name) {}

    void destroy();

private:

    VSNRAY_NOT_COPYABLE(renderbuffer)

};


class texture : public handle<texture>
{
public:

    explicit texture(GLuint name = 0) : handle(name) {}

    void destroy();

private:

    VSNRAY_NOT_COPYABLE(texture)

};


//-------------------------------------------------------------------------------------------------
// factory functions
//

GLuint create_buffer();
GLuint create_framebuffer();
GLuint create_renderbuffer();
GLuint create_texture();


} // gl
} // visionaray


#endif // VSNRAY_GL_HANDLE_H


