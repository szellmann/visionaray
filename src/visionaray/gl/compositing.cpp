// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <visionaray/gl/compositing.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/program.h>
#include <visionaray/gl/util.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// depth compositor private implementation
//

struct depth_compositor::impl
{
#if !defined(VSNRAY_OPENGL_LEGACY)
    impl()
        : prog(glCreateProgram())
        , frag(glCreateShader(GL_FRAGMENT_SHADER))
    {
    }

   ~impl()
    {
        glDetachShader(prog.get(), frag);
        glDeleteShader(frag);
    }

    // GL color texture handle
    gl::texture color_texture;

    // GL color texture handle
    gl::texture depth_texture;

    // The program
    gl::program prog;

    // The fragment shader
    GLuint frag;

    // Uniform location of color texture
    GLint color_loc;

    // Uniform location of depth texture
    GLint depth_loc;

    bool check_shader_compiled() const;
    void enable_program() const;
    void disable_program() const;
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
bool depth_compositor::impl::check_shader_compiled() const
{
    GLint success = GL_FALSE;
    glGetShaderiv(frag, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        GLint length = 0;
        glGetShaderiv(frag, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            std::vector<GLchar> buf( static_cast<size_t>(length) );
            glGetShaderInfoLog( frag, length, &length, buf.data() );
            std::cerr << std::string(buf.data()) << std::endl;
        }

        return false;
    }

    return true;
}

void depth_compositor::impl::enable_program() const
{
    prog.enable();

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());

    glUniform1i(depth_loc, 1);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, depth_texture.get());
}

void depth_compositor::impl::disable_program() const
{
    prog.disable();
}

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
#if !defined(VSNRAY_OPENGL_LEGACY)
    auto source = R"(
        uniform sampler2D color_tex;
    uniform sampler2D depth_tex;

    void main(void)
    {
        gl_FragColor = texture2D(color_tex, gl_TexCoord[0].xy);
        gl_FragDepth = texture2D(depth_tex, gl_TexCoord[0].xy).x;
    }
    )";

    GLint len = static_cast<GLint>(std::strlen(source));

    glShaderSource(impl_->frag, 1, &source, &len);

    glCompileShader(impl_->frag);
    if (!impl_->check_shader_compiled())
    {
        return;
    }

    impl_->prog.reset(glCreateProgram());
    glAttachShader(impl_->prog.get(), impl_->frag);

    impl_->prog.link();
    if (!impl_->prog.check_linked())
    {
        return;
    }

    impl_->color_loc = glGetUniformLocation(impl_->prog.get(), "color_tex");
    impl_->depth_loc = glGetUniformLocation(impl_->prog.get(), "depth_tex");
#endif
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

    impl_->enable_program();

    gl::draw_full_screen_quad();

    impl_->disable_program();


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
    gl::blend_texture(impl_->color_texture.get());
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
