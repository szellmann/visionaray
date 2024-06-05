// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_GL_COMPOSITING_H
#define VSNRAY_COMMON_GL_COMPOSITING_H 1

#include <memory>

#include <GL/glew.h>

#include "../pixel_format.h"

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

    depth_compositor();
    ~depth_compositor();

    void composite_textures() const;

    void display_color_texture() const;

    void setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h);

    void setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h);

    void update_color_texture(
            pixel_format_info   info,
            GLsizei             w,
            GLsizei             h,
            GLvoid const*       data
            ) const;

    void update_depth_texture(
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

#endif // VSNRAY_COMMON_GL_COMPOSITING_H
