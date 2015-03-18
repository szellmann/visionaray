// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_BVH_H
#define VSNRAY_RENDER_BVH_H

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/bvh.h>

namespace visionaray
{

class bvh_outline_renderer
{
public:

    bvh_outline_renderer() = default;

   ~bvh_outline_renderer()
    {
        glDeleteBuffers(1, &vbo_);
    }

    void frame()
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glVertexPointer(3, GL_FLOAT, 0, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_LINES, 0, (GLsizei)(num_vertices));
        glDisableClientState(GL_VERTEX_ARRAY);
    }

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

        num_vertices = vertices.size() / 3;
    }

    GLuint vbo_ = 0;
    size_t num_vertices = 0;

};

} // visionaray

#endif // VSNRAY_RENDER_BVH_H
