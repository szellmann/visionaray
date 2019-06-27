// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_COORDINATES_H
#define VSNRAY_MATH_COORDINATES_H 1

#include "detail/math.h"
#include "config.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Convert from spherical to cartesian coordinates
//

template <typename T>
MATH_FUNC
vector<3, T> spherical_to_cartesian(T const& theta, T const& phi, T const& r)
{
    return vector<3, T>(
        cos(theta) * cos(phi) * r,
        sin(theta) * cos(phi) * r,
        sin(phi) * r
        );
}

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_COORDINATES_H
