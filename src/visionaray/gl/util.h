// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_UTIL_H
#define VSNRAY_GL_UTIL_H

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h> // APIENTRY
#endif
#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/detail/platform.h>
#include <visionaray/math/forward.h>


namespace visionaray
{
namespace gl
{


void init_debug_callback();
void init_ext();

std::string last_error();

void blend_texture(GLuint texture, GLenum sfactor = GL_ONE, GLenum dfactor = GL_ONE_MINUS_SRC_ALPHA);

void blend_pixels(GLsizei w, GLsizei h, GLenum format, GLenum type, GLvoid const* pixels,
    GLenum sfactor = GL_ONE, GLenum dfactor = GL_ONE_MINUS_SRC_ALPHA);

recti viewport();


} // gl
} // visionaray

#endif // VSNRAY_GL_UTIL_H


