// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_COMPOSITING_H
#define VSNRAY_GL_COMPOSITING_H 1

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#else
#include "types.h"
#endif

#include <memory>

#include <visionaray/export.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// GLSL-based depth compositor
//

class depth_compositor
{
public:

    VSNRAY_EXPORT depth_compositor();
    VSNRAY_EXPORT ~depth_compositor();

    VSNRAY_EXPORT void composite_textures() const;

    VSNRAY_EXPORT void display_color_texture() const;

    VSNRAY_EXPORT void setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h);

    VSNRAY_EXPORT void setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h);

    VSNRAY_EXPORT void update_color_texture(
            pixel_format_info   info,
            GLsizei             w,
            GLsizei             h,
            GLvoid const*       data
            ) const;

    VSNRAY_EXPORT void update_depth_texture(
            pixel_format_info   info,
            GLsizei             w,
            GLsizei             h,
            GLvoid const*       data
            ) const;

private:

    struct impl;
    std::unique_ptr<impl> impl_;

    void enable_program() const;

    void disable_program() const;

    void set_texture_params() const;

};


} // gl
} // visionaray

#endif // VSNRAY_GL_COMPOSITING_H
