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
#include <visionaray/detail/macros.h>

#define VSNRAY_OPENGL_LEGACY 0

namespace visionaray
{
namespace gl
{

#if !VSNRAY_OPENGL_LEGACY

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
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
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
#else
{
}
#endif

color_program::~color_program()
{
    prog.detach_shader(vert);
    prog.detach_shader(frag);
}

void color_program::enable(gl::texture const& color_texture) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    prog.enable();

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());
#else
    VSNRAY_UNUSED(color_texture);
#endif
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
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
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
#else
{
}
#endif

depth_program::~depth_program()
{
    prog.detach_shader(frag);
}

void depth_program::enable(
        gl::texture const& color_texture,
        gl::texture const& depth_texture
        ) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
    prog.enable();

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());

    glUniform1i(depth_loc, 1);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, depth_texture.get());
#else
    VSNRAY_UNUSED(color_texture);
    VSNRAY_UNUSED(depth_texture);
#endif
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
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    impl()
        : vertex_buffer(create_buffer())
        , tex_coord_buffer(create_buffer())
    {
        // Store OpenGL state
        GLint array_buffer_binding = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);


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


        // Restore OpenGL state
        glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
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

    void set_texture_params() const;
#else
    pixel_format_info color_info;
    pixel_format_info depth_info;

    GLvoid const* depth_buffer = nullptr;
    GLvoid const* color_buffer = nullptr;

    int width;
    int height;
#endif
#endif
};


#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
void depth_compositor::impl::set_texture_params() const
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
#endif
#endif


//-------------------------------------------------------------------------------------------------
// depth compositor public interface
//

depth_compositor::depth_compositor()
    : impl_(new impl)
{
}

depth_compositor::~depth_compositor() = default;

void depth_compositor::composite_textures() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLint active_texture = GL_TEXTURE0;
    GLuint bound_texture = 0;
    GLboolean blend = glIsEnabled(GL_BLEND);
    GLenum sfactor_rgb = GL_ONE;
    GLenum dfactor_rgb = GL_ONE_MINUS_SRC_ALPHA;
    GLenum sfactor_alpha = GL_ONE;
    GLenum dfactor_alpha = GL_ONE_MINUS_SRC_ALPHA;
    GLint array_buffer_binding = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));
    glGetIntegerv(GL_BLEND_SRC_RGB, reinterpret_cast<GLint*>(&sfactor_rgb));
    glGetIntegerv(GL_BLEND_DST_RGB, reinterpret_cast<GLint*>(&dfactor_rgb));
    glGetIntegerv(GL_BLEND_SRC_ALPHA, reinterpret_cast<GLint*>(&sfactor_alpha));
    glGetIntegerv(GL_BLEND_DST_ALPHA, reinterpret_cast<GLint*>(&dfactor_alpha));
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);


    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

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
    glBindTexture(GL_TEXTURE_2D, bound_texture);
    glBlendFuncSeparate(sfactor_rgb, dfactor_rgb, sfactor_alpha, dfactor_alpha);
    if (blend)
    {
        glEnable(GL_BLEND);
    }
    else
    {
        glDisable(GL_BLEND);
    }
    glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
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
#endif
}

void depth_compositor::display_color_texture() const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLint array_buffer = 0;
    GLint active_texture = GL_TEXTURE0;
    GLuint bound_texture = 0;
    GLboolean blend = glIsEnabled(GL_BLEND);
    GLenum sfactor_rgb = GL_ONE;
    GLenum dfactor_rgb = GL_ONE_MINUS_SRC_ALPHA;
    GLenum sfactor_alpha = GL_ONE;
    GLenum dfactor_alpha = GL_ONE_MINUS_SRC_ALPHA;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));
    glGetIntegerv(GL_BLEND_SRC_RGB, reinterpret_cast<GLint*>(&sfactor_rgb));
    glGetIntegerv(GL_BLEND_DST_RGB, reinterpret_cast<GLint*>(&dfactor_rgb));
    glGetIntegerv(GL_BLEND_SRC_ALPHA, reinterpret_cast<GLint*>(&sfactor_alpha));
    glGetIntegerv(GL_BLEND_DST_ALPHA, reinterpret_cast<GLint*>(&dfactor_alpha));


    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

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
    glBindBuffer(GL_ARRAY_BUFFER, array_buffer);
    glActiveTexture(active_texture);
    glBindTexture(GL_TEXTURE_2D, bound_texture);
    glBlendFuncSeparate(sfactor_rgb, dfactor_rgb, sfactor_alpha, dfactor_alpha);
    if (blend)
    {
        glEnable(GL_BLEND);
    }
    else
    {
        glDisable(GL_BLEND);
    }
#else
    gl::blend_pixels(
            impl_->width,
            impl_->height,
            impl_->color_info.format,
            impl_->color_info.type,
            impl_->color_buffer
            );
#endif
#endif
}

void depth_compositor::setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLuint bound_texture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));


    impl_->color_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);


    // Restore OpenGL state
    glBindTexture(GL_TEXTURE_2D, bound_texture);
#else
    impl_->color_info = info;
    impl_->width = w;
    impl_->height = h;
#endif
#else
    VSNRAY_UNUSED(info);
    VSNRAY_UNUSED(w);
    VSNRAY_UNUSED(h);
#endif
}

void depth_compositor::setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLuint bound_texture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));


    impl_->depth_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);


    // Restore OpenGL state
    glBindTexture(GL_TEXTURE_2D, bound_texture);
#else
    impl_->depth_info = info;
    impl_->width = w;
    impl_->height = h;
#endif
#else
    VSNRAY_UNUSED(info);
    VSNRAY_UNUSED(w);
    VSNRAY_UNUSED(h);
#endif
}

void depth_compositor::update_color_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLuint bound_texture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));


    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    gl::update_texture( info, w, h, data );


    // Restore OpenGL state
    glBindTexture(GL_TEXTURE_2D, bound_texture);
#else
    impl_->color_info = info;
    impl_->width = w;
    impl_->height = h;
    impl_->color_buffer = data;
#endif
#else
    VSNRAY_UNUSED(info);
    VSNRAY_UNUSED(w);
    VSNRAY_UNUSED(h);
    VSNRAY_UNUSED(data);
#endif
}

void depth_compositor::update_depth_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
#if VSNRAY_HAVE_GLEW || VSNRAY_HAVE_OPENGLES
#if !VSNRAY_OPENGL_LEGACY
    // Store OpenGL state
    GLuint bound_texture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, reinterpret_cast<GLint*>(&bound_texture));


    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    gl::update_texture( info, w, h, data );


    // Restore OpenGL state
    glBindTexture(GL_TEXTURE_2D, bound_texture);
#else
    impl_->depth_info = info;
    impl_->width = w;
    impl_->height = h;
    impl_->depth_buffer = data;
#endif
#else
    VSNRAY_UNUSED(info);
    VSNRAY_UNUSED(w);
    VSNRAY_UNUSED(h);
    VSNRAY_UNUSED(data);
#endif
}

} // gl
} // visionaray
