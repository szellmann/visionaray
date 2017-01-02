// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_COMPOSITING_H
#define VSNRAY_GL_COMPOSITING_H 1

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/gl/handle.h>
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

    // GL color texture handle
    gl::texture color_texture_;

    // GL color texture handle
    gl::texture depth_texture_;

    // The program
    GLuint program_;

    // The fragment shader
    GLuint frag_;

    // Uniform location of color texture
    GLint color_loc_;

    // Uniform location of depth texture
    GLint depth_loc_;


    bool check_shader_compiled() const;

    bool check_program_linked() const;

    void enable_program() const;

    void disable_program() const;

    void set_texture_params() const;

};


} // gl
} // visionaray

#endif // VSNRAY_GL_COMPOSITING_H
