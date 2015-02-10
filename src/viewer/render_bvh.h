// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_RENDER_BVH_H
#define VSNRAY_VIEWER_RENDER_BVH_H

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <visionaray/bvh.h>

namespace visionaray
{

template <typename BVH>
void render_bvh(BVH const& b)
{

    std::vector<float> vertices;
    for (auto const& n : b.nodes())
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
    }

    glVertexPointer(3, GL_FLOAT, 0, vertices.data());
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_LINES, 0, (GLsizei)(vertices.size() / 3));
    glDisableClientState(GL_VERTEX_ARRAY);

}

} // visionaray

#endif // VSNRAY_VIEWER_RENDER_BVH_H


