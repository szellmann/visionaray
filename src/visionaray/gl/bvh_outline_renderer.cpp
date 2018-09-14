// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <visionaray/gl/bvh_outline_renderer.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/program.h>
#include <visionaray/gl/shader.h>

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct bvh_outline_renderer::impl
{
    gl::buffer  vertex_buffer;
    gl::program prog;
    gl::shader  vert;
    gl::shader  frag;
    GLuint      view_loc;
    GLuint      proj_loc;
    GLuint      vertex_loc;
};


//-------------------------------------------------------------------------------------------------
// BVH outline renderer OpenGL implementation
//

bvh_outline_renderer::bvh_outline_renderer()
    : impl_(new impl)
{
}

bvh_outline_renderer::~bvh_outline_renderer() = default;

void bvh_outline_renderer::frame(mat4 const& view, mat4 const& proj) const
{
    // Store OpenGL state
    GLint array_buffer_binding = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);
    GLboolean depth_clamp_enabled = glIsEnabled(GL_DEPTH_CLAMP);


    glEnable(GL_DEPTH_CLAMP);

    impl_->prog.enable();

    glUniformMatrix4fv(impl_->view_loc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(impl_->proj_loc, 1, GL_FALSE, proj.data());

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vertex_buffer.get());
    glVertexAttribPointer(impl_->vertex_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(impl_->vertex_loc);

    glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices_));

    glDisableVertexAttribArray(impl_->vertex_loc);

    impl_->prog.disable();


    // Restore OpenGL state
    if (depth_clamp_enabled)
    {
        glEnable(GL_DEPTH_CLAMP);
    }
    else
    {
        glDisable(GL_DEPTH_CLAMP);
    }
    glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
}

void bvh_outline_renderer::destroy()
{
    impl_->vertex_buffer.destroy();

    if (impl_->prog.check_attached(impl_->vert))
    {
        impl_->prog.detach_shader(impl_->vert);
    }

    if (impl_->prog.check_attached(impl_->frag))
    {
        impl_->prog.detach_shader(impl_->frag);
    }
}

bool bvh_outline_renderer::init_gl(float const* data, size_t size)
{
    // Store OpenGL state
    GLint array_buffer_binding = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);



    // Setup shaders

    impl_->vert.reset(glCreateShader(GL_VERTEX_SHADER));
    impl_->vert.set_source(R"(
        attribute vec3 vertex;
        uniform mat4 view;
        uniform mat4 proj;

        void main(void)
        {
            gl_Position = proj * view * vec4(vertex, 1.0);
        }
        )");
    impl_->vert.compile();

    if (!impl_->vert.check_compiled())
    {
        return false;
    }

    impl_->frag.reset(glCreateShader(GL_FRAGMENT_SHADER));
    impl_->frag.set_source(R"(
        void main(void)
        {
            gl_FragColor = vec4(1.0);
        }        
        )");
    impl_->frag.compile();

    if (!impl_->frag.check_compiled())
    {
        return false;
    }

    impl_->prog.reset(glCreateProgram());
    impl_->prog.attach_shader(impl_->vert);
    impl_->prog.attach_shader(impl_->frag);

    impl_->prog.link();

    if (!impl_->prog.check_linked())
    {
        return false;
    }

    impl_->vertex_loc = glGetAttribLocation(impl_->prog.get(), "vertex");
    impl_->view_loc   = glGetUniformLocation(impl_->prog.get(), "view");
    impl_->proj_loc   = glGetUniformLocation(impl_->prog.get(), "proj");


    // Setup vbo
    impl_->vertex_buffer.reset(create_buffer());

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vertex_buffer.get());
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);


    // Restore OpenGL state
    glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);

    return true;
}

} // gl
} // visionaray
