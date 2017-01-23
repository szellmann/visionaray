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

void bvh_outline_renderer::frame() const
{
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_.get());
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices_));
    glDisableClientState(GL_VERTEX_ARRAY);
}

void bvh_outline_renderer::destroy()
{
    vertex_buffer_.destroy();
}

void bvh_outline_renderer::init_vbo(float const* data, size_t size)
{
    vertex_buffer_.reset(create_buffer());

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_.get());
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

} // gl
} // visionaray
