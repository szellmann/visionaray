// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_POINT_LIGHT_H
#define VSNRAY_POINT_LIGHT_H

#include "math/math.h"

namespace visionaray
{

template <typename T>
struct point_light
{
    typedef T scalar_type;
    typedef vector<3, scalar_type> vec_type;

    vec_type position_;
};

} // visionaray

#endif // VSNRAY_POINT_LIGHT_H


