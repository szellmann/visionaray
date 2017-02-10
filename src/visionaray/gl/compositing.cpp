// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <visionaray/gl/compositing.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/program.h>
#include <visionaray/gl/shader.h>
#include <visionaray/gl/util.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{
namespace gl
{

#if !defined(VSNRAY_OPENGL_LEGACY)

//-------------------------------------------------------------------------------------------------
// Shader program to display color texture w/o depth compositing
//

struct color_program
{
    color_program();
   ~color_program();

    // The program
    gl::program prog;

    // The fragment shader
    gl::shader frag;

    // Uniform location of color texture
    GLint color_loc;

    void enable(gl::texture const& color_texture) const;
    void disable() const;
};


//-------------------------------------------------------------------------------------------------
// color program implementation
//

color_program::color_program()
    : prog(glCreateProgram())
    , frag(glCreateShader(GL_FRAGMENT_SHADER))
{
    frag.set_source(R"(
        uniform sampler2D color_tex;

        void main(void)
        {
            gl_FragColor = texture2D(color_tex, gl_TexCoord[0].xy);
        }
        )");

    frag.compile();
    if (!frag.check_compiled())
    {
        return;
    }

    prog.attach_shader(frag);

    prog.link();
    if (!prog.check_linked())
    {
        return;
    }

    color_loc = glGetUniformLocation(prog.get(), "color_tex");
}

color_program::~color_program()
{
    prog.detach_shader(frag);
}

void color_program::enable(gl::texture const& color_texture) const
{
    prog.enable();

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());
}

void color_program::disable() const
{
    prog.disable();
}


//-------------------------------------------------------------------------------------------------
// Shader program to composite depth textures
//

struct depth_program
{
    depth_program();
   ~depth_program();

    // The program
    gl::program prog;

    // The fragment shader
    gl::shader frag;

    // Uniform location of color texture
    GLint color_loc;

    // Uniform location of depth texture
    GLint depth_loc;

    void enable(gl::texture const& color_texture, gl::texture const& depth_texture) const;
    void disable() const;
};


//-------------------------------------------------------------------------------------------------
// depth program implementation
//

depth_program::depth_program()
    : prog(glCreateProgram())
    , frag(glCreateShader(GL_FRAGMENT_SHADER))
{
    frag.set_source(R"(
        uniform sampler2D color_tex;
        uniform sampler2D depth_tex;

        void main(void)
        {
            gl_FragColor = texture2D(color_tex, gl_TexCoord[0].xy);
            gl_FragDepth = texture2D(depth_tex, gl_TexCoord[0].xy).x;
        }
        )");

    frag.compile();
    if (!frag.check_compiled())
    {
        return;
    }

    prog.attach_shader(frag);

    prog.link();
    if (!prog.check_linked())
    {
        return;
    }

    color_loc = glGetUniformLocation(prog.get(), "color_tex");
    depth_loc = glGetUniformLocation(prog.get(), "depth_tex");
}

depth_program::~depth_program()
{
    prog.detach_shader(frag);
}

void depth_program::enable(
        gl::texture const& color_texture,
        gl::texture const& depth_texture
        ) const
{
    prog.enable();

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());

    glUniform1i(depth_loc, 1);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, depth_texture.get());
}

void depth_program::disable() const
{
    prog.disable();
}

#endif // !VSNRAY_OPENGL_LEGACY


//-------------------------------------------------------------------------------------------------
// depth compositor private implementation
//

struct depth_compositor::impl
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    // Shader program to only display color texture w/o depth compositing
    color_program color_prog;

    // Shader program for depth compositing
    depth_program depth_prog;

    // GL color texture handle
    gl::texture color_texture;

    // GL color texture handle
    gl::texture depth_texture;

    void set_texture_params() const;
#else
    pixel_format_info color_info;
    pixel_format_info depth_info;

    GLvoid const* depth_buffer = nullptr;
    GLvoid const* color_buffer = nullptr;

    int width;
    int height;
#endif
};


#if !defined(VSNRAY_OPENGL_LEGACY)
void depth_compositor::impl::set_texture_params() const
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
#endif


//-------------------------------------------------------------------------------------------------
// depth compositor public interface
//

depth_compositor::depth_compositor()
    : impl_(new impl)
{
}

depth_compositor::~depth_compositor()
{
}

void depth_compositor::composite_textures() const
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    // Store OpenGL state
    GLint active_texture = GL_TEXTURE0;
    GLboolean depth_test = GL_FALSE;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
    glGetBooleanv(GL_DEPTH_TEST, &depth_test);


    glEnable(GL_DEPTH_TEST);

    impl_->depth_prog.enable(impl_->color_texture, impl_->depth_texture);

    gl::draw_full_screen_quad();

    impl_->depth_prog.disable();


    // Restore OpenGL state
    glActiveTexture(active_texture);
    if (depth_test)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }
#else
    glPushAttrib( GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_ENABLE_BIT );

    glEnable(GL_STENCIL_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glStencilFunc(GL_ALWAYS, 1, 1);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    gl::blend_pixels(
            impl_->width,
            impl_->height,
            impl_->depth_info.format,
            impl_->depth_info.type,
            impl_->depth_buffer
            );

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 1);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glDisable(GL_DEPTH_TEST);

    gl::blend_pixels(
            impl_->width,
            impl_->height,
            impl_->color_info.format,
            impl_->color_info.type,
            impl_->color_buffer
            );

    glPopAttrib();
#endif
}

void depth_compositor::display_color_texture() const
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    // Store OpenGL state
    GLint active_texture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);


    impl_->color_prog.enable(impl_->color_texture);

    gl::draw_full_screen_quad();

    impl_->color_prog.disable();


    // Restore OpenGL state
    glActiveTexture(active_texture);
#else
    gl::blend_pixels(
            impl_->width,
            impl_->height,
            impl_->color_info.format,
            impl_->color_info.type,
            impl_->color_buffer
            );
#endif
}

void depth_compositor::setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    impl_->color_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);
#else
    impl_->color_info = info;
    impl_->width = w;
    impl_->height = h;
#endif
}

void depth_compositor::setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    impl_->depth_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);
#else
    impl_->depth_info = info;
    impl_->width = w;
    impl_->height = h;
#endif
}

void depth_compositor::update_color_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    gl::update_texture( info, w, h, data );
#else
    impl_->color_info = info;
    impl_->width = w;
    impl_->height = h;
    impl_->color_buffer = data;
#endif
}

void depth_compositor::update_depth_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    gl::update_texture( info, w, h, data );
#else
    impl_->depth_info = info;
    impl_->width = w;
    impl_->height = h;
    impl_->depth_buffer = data;
#endif
}

} // gl
} // visionaray
