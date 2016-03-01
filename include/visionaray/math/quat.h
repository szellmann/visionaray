// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_QUAT_H
#define VSNRAY_MATH_QUAT_H 1

#include "vector.h"

namespace MATH_NAMESPACE
{

class quat
{
public:

    float w;
    float x;
    float y;
    float z;

    quat();
    quat(float w, float x, float y, float z);
    quat(float w, vec3 const& v);

    static quat identity();
    static quat rotation(vec3 const& from, vec3 const& to);

};

} // MATH_NAMESPACE

#include "detail/quat.inl"

#endif // VSNRAY_MATH_QUAT_H
