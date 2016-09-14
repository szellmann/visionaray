// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_RENDER_BVH_H
#define VSNRAY_DETAIL_RENDER_BVH_H 1

#include "platform.h"

#include <vector>

#include <GL/glew.h>

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/bvh.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// OpenGL BVH outline renderer
// Call init() and destroy() with a valid OpenGL context!
//

class bvh_outline_renderer
{
public:

    //-------------------------------------------------------------------------
    // Render BVH outlines
    //

    void frame() const
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glVertexPointer(3, GL_FLOAT, 0, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices_));
        glDisableClientState(GL_VERTEX_ARRAY);
    }


    //-------------------------------------------------------------------------
    // init()
    // Call with a valid OpenGL context!
    //

    template <typename BVH>
    void init(BVH const& b)
    {
        glDeleteBuffers(1, &vbo_);
        glGenBuffers(1, &vbo_);

        std::vector<float> vertices;
        traverse_depth_first(b, [&](typename BVH::node_type const& n)
        {
            auto box = n.bbox;

            auto ilist =
            {
                box.min.x, box.min.y, box.min.z,
                box.max.x, box.min.y, box.min.z,

                box.max.x, box.min.y, box.min.z,
                box.max.x, box.max.y, box.min.z,

                box.max.x, box.max.y, box.min.z,
                box.min.x, box.max.y, box.min.z,

                box.min.x, box.max.y, box.min.z,
                box.min.x, box.min.y, box.min.z,

                box.min.x, box.min.y, box.max.z,
                box.max.x, box.min.y, box.max.z,

                box.max.x, box.min.y, box.max.z,
                box.max.x, box.max.y, box.max.z,

                box.max.x, box.max.y, box.max.z,
                box.min.x, box.max.y, box.max.z,

                box.min.x, box.max.y, box.max.z,
                box.min.x, box.min.y, box.max.z,

                box.min.x, box.min.y, box.min.z,
                box.min.x, box.min.y, box.max.z,

                box.max.x, box.min.y, box.min.z,
                box.max.x, box.min.y, box.max.z,

                box.max.x, box.max.y, box.min.z,
                box.max.x, box.max.y, box.max.z,

                box.min.x, box.max.y, box.min.z,
                box.min.x, box.max.y, box.max.z
            };

            vertices.insert(vertices.end(), ilist.begin(), ilist.end());
        });


        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        num_vertices_ = vertices.size() / 3;
    }


    //-------------------------------------------------------------------------
    // destroy()
    // Call while OpenGL context is still valid!
    //

    void destroy()
    {
        glDeleteBuffers(1, &vbo_);
    }


private:

    GLuint vbo_ = 0;
    size_t num_vertices_ = 0;

};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_RENDER_BVH_H
