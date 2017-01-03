// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <GL/glew.h>

#include <visionaray/gl/compositing.h>
#include <visionaray/gl/handle.h>
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
    impl()
        : program(glCreateProgram())
        , frag(glCreateShader(GL_FRAGMENT_SHADER))
    {
    }

   ~impl()
    {
        glDetachShader(program, frag);
        glDeleteShader(frag);
        glDeleteProgram(program);
    }

    // GL color texture handle
    gl::texture color_texture;

    // GL color texture handle
    gl::texture depth_texture;

    // The program
    GLuint program;

    // The fragment shader
    GLuint frag;

    // Uniform location of color texture
    GLint color_loc;

    // Uniform location of depth texture
    GLint depth_loc;

    bool check_shader_compiled() const;
    bool check_program_linked() const;
    void enable_program() const;
    void disable_program() const;
    void set_texture_params() const;
};


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

bool depth_compositor::impl::check_program_linked() const
{
    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            std::vector<GLchar> buf( static_cast<size_t>(length) );
            glGetProgramInfoLog( program, length, &length, buf.data() );
            std::cerr << std::string(buf.data()) << std::endl;
        }

        return false;
    }

    return true;
}

void depth_compositor::impl::enable_program() const
{
    glUseProgram(program);

    glUniform1i(color_loc, 0);
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, color_texture.get());

    glUniform1i(depth_loc, 1);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, depth_texture.get());
}

void depth_compositor::impl::disable_program() const
{
    glUseProgram(0);
}

void depth_compositor::impl::set_texture_params() const
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


//-------------------------------------------------------------------------------------------------
// depth compositor public interface
//

depth_compositor::depth_compositor()
    : impl_(new impl)
{
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

    impl_->program = glCreateProgram();
    glAttachShader(impl_->program, impl_->frag);

    glLinkProgram(impl_->program);
    if (!impl_->check_program_linked())
    {
        return;
    }

    impl_->color_loc = glGetUniformLocation(impl_->program, "color_tex");
    impl_->depth_loc = glGetUniformLocation(impl_->program, "depth_tex");
}

depth_compositor::~depth_compositor()
{
}

void depth_compositor::composite_textures() const
{
    glPushAttrib(GL_TEXTURE_BIT | GL_ENABLE_BIT);

    glEnable(GL_DEPTH_TEST);

    impl_->enable_program();

    gl::draw_full_screen_quad();

    impl_->disable_program();

    glPopAttrib();
}

void depth_compositor::display_color_texture() const
{
    gl::blend_texture(impl_->color_texture.get());
}

void depth_compositor::setup_color_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
    glPushAttrib(GL_TEXTURE_BIT);

    impl_->color_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);

    glPopAttrib();
}

void depth_compositor::setup_depth_texture(pixel_format_info info, GLsizei w, GLsizei h)
{
    glPushAttrib(GL_TEXTURE_BIT);

    impl_->depth_texture.reset( create_texture() );

    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    impl_->set_texture_params();
    alloc_texture(info, w, h);

    glPopAttrib();
}

void depth_compositor::update_color_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
    glPushAttrib(GL_TEXTURE_BIT);

    glBindTexture(GL_TEXTURE_2D, impl_->color_texture.get());

    gl::update_texture( info, w, h, data );

    glPopAttrib();
}

void depth_compositor::update_depth_texture(
        pixel_format_info   info,
        GLsizei             w,
        GLsizei             h,
        GLvoid const*       data
        ) const
{
    glPushAttrib(GL_TEXTURE_BIT);

    glBindTexture(GL_TEXTURE_2D, impl_->depth_texture.get());

    gl::update_texture( info, w, h, data );

    glPopAttrib();
}

} // gl
} // visionaray
