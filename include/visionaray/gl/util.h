// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_UTIL_H
#define VSNRAY_GL_UTIL_H 1

#include <string>
#include <vector>

#include <visionaray/detail/platform.h>

#ifdef _WIN32
#ifndef APIENTRY
#include <windows.h> // APIENTRY
#undef min
#undef max
#endif
#endif
#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/math/forward.h>
#include <visionaray/pixel_format.h>


namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// Flags used in debug callback function
//

enum debug_severity
{
    Notification,
    Low,
    Medium,
    High
};

enum debug_type
{
    // can be combined bitwise

    Error               = 0x00000001,
    DeprecatedBehavior  = 0x00000002,
    UndefinedBehavior   = 0x00000004,
    Portability         = 0x00000008,
    Performance         = 0x00000010,
    Other               = 0x00000020,

    None                = 0x00000000
};


//-------------------------------------------------------------------------------------------------
// Debug parameters passed to debug callback function
//
//  param severity
//      filter messages by severity
//
//  param types
//      overrides param severity
//      whitelist several debug message types
//      bitwise combination of debug_type values
//

struct debug_params
{
    debug_severity  severity    = debug_severity::High;
    debug_type      types       = debug_type::None;
};

void init_debug_callback(debug_params params = debug_params());

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
