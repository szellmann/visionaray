// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_UTIL_H
#define VSNRAY_GL_UTIL_H 1

#include <string>

#include <visionaray/detail/platform.h>

#ifdef _WIN32
#ifndef APIENTRY
#include <windows.h> // APIENTRY
#undef min
#undef max
#endif
#endif

#include <visionaray/math/forward.h>
#include <visionaray/pixel_format.h>

#include "types.h"


namespace visionaray
{
namespace gl
{

std::string last_error();

void alloc_texture(pixel_format_info info, GLsizei w, GLsizei h);

void update_texture(
        pixel_format_info   info,
        GLsizei             x,
        GLsizei             y,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       pixels
        );

void update_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       pixels
        );

void draw_full_screen_quad();

void blend_texture(
        GLuint          texture,
        GLenum          sfactor     = GL_ONE,
        GLenum          dfactor     = GL_ONE_MINUS_SRC_ALPHA
        );

void blend_pixels(
        GLsizei         w,
        GLsizei         h,
        GLenum          format,
        GLenum          type,
        GLvoid const*   pixels,
        GLenum          sfactor     = GL_ONE,
        GLenum          dfactor     = GL_ONE_MINUS_SRC_ALPHA
        );

recti viewport();

} // gl
} // visionaray

#endif // VSNRAY_GL_UTIL_H
