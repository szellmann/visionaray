// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_BVH_OUTLINE_RENDERER_H
#define VSNRAY_GL_BVH_OUTLINE_RENDERER_H 1

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#elif VSNRAY_HAVE_OPENGLES
#include <GLES2/gl2.h>
#endif

#include <vector>

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

public:

    // Render BVH outlines
    void frame() const;

    // Call init() with a valid OpenGL context!
    template <typename BVH>
    void init(BVH const& b, display_config config = display_config())
    {
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


        init_vbo(vertices.data(), vertices.size() * sizeof(float));

        num_vertices_ = vertices.size() / 3;
    }


    // Call destroy() while OpenGL context is still valid!
    void destroy();

private:

    GLuint vbo_ = 0;
    size_t num_vertices_ = 0;

    // init vbo, pointer to vertices, buffer size in bytes
    void init_vbo(float const* data, size_t size);

};

} // gl
} // visionaray

#endif // VSNRAY_GL_BVH_OUTLINE_RENDERER_H
