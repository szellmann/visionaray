// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_BVH_OUTLINE_RENDERER_H
#define VSNRAY_GL_BVH_OUTLINE_RENDERER_H 1

#include <vector>

#include <GL/glew.h>

#include "../detail/platform.h"

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/bvh.h>

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// OpenGL BVH outline renderer
// Call init() and destroy() with a valid OpenGL context!
//

class bvh_outline_renderer
{
public:

    //-------------------------------------------------------------------------
    // Configuration
    //

    enum display_filter
    {
        Full = 0,   // display the whole BVH
        Leaves,     // display only leave nodes
//      Level       // display only nodes at a certain level
    };

    struct display_config
    {
        display_config()
            : filter(Full)
//          , level(-1)
        {
        }

        display_filter filter;
//      int            level ;
    };


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
    void init(BVH const& b, display_config config = display_config())
    {
        glDeleteBuffers(1, &vbo_);
        glGenBuffers(1, &vbo_);

        std::vector<float> vertices;
        auto func =  [&](typename BVH::node_type const& n)
        {
            auto box = n.get_bounds();

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
        };

        if (config.filter == Full)
        {
            traverse_depth_first(b, func);
        }
        else if (config.filter == Leaves)
        {
            traverse_leaves(b, func);
        }


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

} // gl
} // visionaray

#endif // VSNRAY_GL_BVH_OUTLINE_RENDERER_H
