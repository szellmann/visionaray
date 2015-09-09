// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ARCBALL_H
#define VSNRAY_ARCBALL_H 1

#include <visionaray/math/math.h>

namespace visionaray
{

struct arcball
{
    arcball()
        : radius(1.0f)
        , down_pos(0.0f)
        , rotation(quat::identity())
        , down_rotation(quat::identity())
    {

    }

    vec3 project(int x, int y, recti const& r);

    float radius;
    vec3 down_pos;
    quat rotation;
    quat down_rotation;

};

} // visionaray

#endif // VSNRAY_ARCBALL_H
