// This file is distributed under the MIT license.
// See the LICENSE file for details.

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
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices_));
    glDisableClientState(GL_VERTEX_ARRAY);
}

void bvh_outline_renderer::destroy()
{
    glDeleteBuffers(1, &vbo_);
}

void bvh_outline_renderer::init_vbo(float const* data, size_t size)
{
    glDeleteBuffers(1, &vbo_);
    glGenBuffers(1, &vbo_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

} // gl
} // visionaray
