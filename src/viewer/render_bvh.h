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
namespace detail
{

void render(aabb const& box)
{
    glBegin(GL_LINES);

        glVertex3f(box.min.x, box.min.y, box.min.z);
        glVertex3f(box.max.x, box.min.y, box.min.z);

        glVertex3f(box.max.x, box.min.y, box.min.z);
        glVertex3f(box.max.x, box.max.y, box.min.z);

        glVertex3f(box.max.x, box.max.y, box.min.z);
        glVertex3f(box.min.x, box.max.y, box.min.z);

        glVertex3f(box.min.x, box.max.y, box.min.z);
        glVertex3f(box.min.x, box.min.y, box.min.z);

        glVertex3f(box.min.x, box.min.y, box.max.z);
        glVertex3f(box.max.x, box.min.y, box.max.z);

        glVertex3f(box.max.x, box.min.y, box.max.z);
        glVertex3f(box.max.x, box.max.y, box.max.z);

        glVertex3f(box.max.x, box.max.y, box.max.z);
        glVertex3f(box.min.x, box.max.y, box.max.z);

        glVertex3f(box.min.x, box.max.y, box.max.z);
        glVertex3f(box.min.x, box.min.y, box.max.z);


        glVertex3f(box.min.x, box.min.y, box.min.z);
        glVertex3f(box.min.x, box.min.y, box.max.z);

        glVertex3f(box.max.x, box.min.y, box.min.z);
        glVertex3f(box.max.x, box.min.y, box.max.z);

        glVertex3f(box.max.x, box.max.y, box.min.z);
        glVertex3f(box.max.x, box.max.y, box.max.z);

        glVertex3f(box.min.x, box.max.y, box.min.z);
        glVertex3f(box.min.x, box.max.y, box.max.z);

    glEnd();
}

} // detail

template <typename P>
void render(bvh<P> const& b)
{

    for (auto const& n : b.nodes_vector())
    {
        detail::render(n.bbox);
    }

}

} // visionaray

#endif // VSNRAY_VIEWER_RENDER_BVH_H


