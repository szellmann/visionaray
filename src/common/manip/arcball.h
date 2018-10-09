// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_ARCBALL_H
#define VSNRAY_COMMON_MANIP_ARCBALL_H 1

#include <visionaray/math/forward.h>
#include <visionaray/math/quaternion.h>
#include <visionaray/math/rectangle.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

class arcball
{
public:

    arcball() = default;
    explicit arcball(float r);

    vec3 project(int x, int y, recti const& viewport);

    float radius = 1.0f;
    vec3 down_pos = vec3(0.0f);
    quat rotation = quat::identity();
    quat down_rotation = quat::identity();

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_ARCBALL_H
