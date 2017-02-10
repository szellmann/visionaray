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

    // The vertex shader
    gl::shader vert;

    // The fragment shader
    gl::shader frag;

    // Attribute location of vertex
    GLint vertex_loc;

    // Attribute location of texture coordinate
    GLint tex_coord_loc;

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
    , vert(glCreateShader(GL_VERTEX_SHADER))
    , frag(glCreateShader(GL_FRAGMENT_SHADER))
{
    vert.set_source(R"(
        attribute vec2 vertex;
        attribute vec2 tex_coord;

        varying vec2 uv;

        void main(void)
        {
            gl_Position = vec4(vertex, 0.0, 1.0);
            uv = tex_coord;
        }
        )");

    vert.compile();
    if (!vert.check_compiled())
    {
        return;
    }

    frag.set_source(R"(
        varying vec2 uv;
        uniform sampler2D color_tex;

        void main(void)
        {
            gl_FragColor = texture2D(color_tex, uv);
        }
        )");

    frag.compile();
    if (!frag.check_compiled())
    {
        return;
    }

    prog.attach_shader(vert);
    prog.attach_shader(frag);

    prog.link();
    if (!prog.check_linked())
    {
        return;
    }

    vertex_loc    = glGetAttribLocation(prog.get(), "vertex");
    tex_coord_loc = glGetAttribLocation(prog.get(), "tex_coord");
    color_loc     = glGetUniformLocation(prog.get(), "color_tex");
}

color_program::~color_program()
{
    prog.detach_shader(vert);
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

    // The vertex shader
    gl::shader vert;

    // The fragment shader
    gl::shader frag;

    // Attribute location of vertex
    GLint vertex_loc;

    // Attribute location of texture coordinate
    GLint tex_coord_loc;

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
    , vert(glCreateShader(GL_VERTEX_SHADER))
    , frag(glCreateShader(GL_FRAGMENT_SHADER))
{
    vert.set_source(R"(
        attribute vec2 vertex;
        attribute vec2 tex_coord;

        varying vec2 uv;

        void main(void)
        {
            gl_Position = vec4(vertex, 0.0, 1.0);
            uv = tex_coord;
        }
        )");

    vert.compile();
    if (!vert.check_compiled())
    {
        return;
    }

    frag.set_source(R"(
        varying vec2 uv;
        uniform sampler2D color_tex;
        uniform sampler2D depth_tex;

        void main(void)
        {
            gl_FragColor = texture2D(color_tex, uv);
            gl_FragDepth = texture2D(depth_tex, uv).x;
        }
        )");

    frag.compile();
    if (!frag.check_compiled())
    {
        return;
    }

    prog.attach_shader(vert);
    prog.attach_shader(frag);

    prog.link();
    if (!prog.check_linked())
    {
        return;
    }

    vertex_loc    = glGetAttribLocation(prog.get(), "vertex");
    tex_coord_loc = glGetAttribLocation(prog.get(), "tex_coord");
    color_loc     = glGetUniformLocation(prog.get(), "color_tex");
    depth_loc     = glGetUniformLocation(prog.get(), "depth_tex");
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
    impl()
        : vertex_buffer(create_buffer())
        , tex_coord_buffer(create_buffer())
    {
        GLfloat verts[8] = {
                -1.0f, -1.0f,
                 1.0f, -1.0f,
                 1.0f,  1.0f,
                -1.0f,  1.0f
                };

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.get());
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), verts, GL_STATIC_DRAW);

        GLfloat tex_coords[8] = {
                0.0f, 0.0f,
                1.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 1.0f
                };

        glBindBuffer(GL_ARRAY_BUFFER, tex_coord_buffer.get());
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), tex_coords, GL_STATIC_DRAW);
    }

    // Shader program to only display color texture w/o depth compositing
    color_program color_prog;

    // Shader program for depth compositing
    depth_program depth_prog;

    // Quad vertex buffer
    gl::buffer vertex_buffer;

    // Quad texture coordinate buffer
    gl::buffer tex_coord_buffer;

    // GL color texture handle
    gl::texture color_texture;

    // GL color texture handle
    gl::texture depth_texture;
#else
    pixel_format_info color_info;
    pixel_format_info depth_info;

    GLvoid const* depth_buffer = nullptr;
    GLvoid const* color_buffer = nullptr;

    int width;
    int height;
#endif
};


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

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vertex_buffer.get());
    glVertexAttribPointer(impl_->depth_prog.vertex_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(impl_->depth_prog.vertex_loc);

    glBindBuffer(GL_ARRAY_BUFFER, impl_->tex_coord_buffer.get());
    glVertexAttribPointer(impl_->depth_prog.tex_coord_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(impl_->depth_prog.tex_coord_loc);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableVertexAttribArray(impl_->depth_prog.tex_coord_loc);
    glDisableVertexAttribArray(impl_->depth_prog.vertex_loc);

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

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vertex_buffer.get());
    glVertexAttribPointer(impl_->color_prog.vertex_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(impl_->color_prog.vertex_loc);

    glBindBuffer(GL_ARRAY_BUFFER, impl_->tex_coord_buffer.get());
    glVertexAttribPointer(impl_->color_prog.tex_coord_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(impl_->color_prog.tex_coord_loc);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableVertexAttribArray(impl_->color_prog.tex_coord_loc);
    glDisableVertexAttribArray(impl_->color_prog.vertex_loc);

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
