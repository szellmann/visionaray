// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <visionaray/gl/bvh_outline_renderer.h>

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// BVH outline renderer OpenGL implementation
//

void bvh_outline_renderer::frame(mat4 const& view, mat4 const& proj) const
{
    prog_.enable();

    glUniformMatrix4fv(view_loc_, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(proj_loc_, 1, GL_FALSE, proj.data());

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_.get());
    glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices_));

    prog_.disable();
}

void bvh_outline_renderer::destroy()
{
    vertex_buffer_.destroy();
    prog_.detach_shader(vert_);
    prog_.detach_shader(frag_);
}

bool bvh_outline_renderer::init_gl(float const* data, size_t size)
{
    // Setup shaders

    vert_.reset(glCreateShader(GL_VERTEX_SHADER));
    vert_.set_source(R"(
        attribute vec3 vertex;
        uniform mat4 view;
        uniform mat4 proj;

        void main(void)
        {
            gl_Position = proj * view * vec4(vertex, 1.0);
        }
        )");
    vert_.compile();

    if (!vert_.check_compiled())
    {
        return false;
    }

    frag_.reset(glCreateShader(GL_FRAGMENT_SHADER));
    frag_.set_source(R"(
        void main(void)
        {
            gl_FragColor = vec4(1.0);
        }        
        )");
    frag_.compile();

    if (!frag_.check_compiled())
    {
        return false;
    }

    prog_.reset(glCreateProgram());
    prog_.attach_shader(vert_);
    prog_.attach_shader(frag_);

    prog_.link();

    if (!prog_.check_linked())
    {
        return false;
    }

    vertex_loc_     = glGetAttribLocation(prog_.get(), "vertex");
    view_loc_  = glGetUniformLocation(prog_.get(), "view");
    proj_loc_ = glGetUniformLocation(prog_.get(), "proj");


    // Setup vbo
    vertex_buffer_.reset(create_buffer());

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_.get());
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    glVertexAttribPointer(vertex_loc_, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vertex_loc_);

    return true;
}

} // gl
} // visionaray
