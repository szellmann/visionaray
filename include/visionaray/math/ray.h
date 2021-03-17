// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_RAY_H
#define VSNRAY_MATH_RAY_H 1

#include "config.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T>
class basic_ray
{
public:

    typedef T scalar_type;
    typedef vector<3, T> vec_type;

public:

    vector<3, T> ori;
    vector<3, T> dir;

    T tmin;
    T tmax;

    basic_ray() = default;

    // Constructor with origin and direction, tmin is 0.0 and tmax is
    // numeric_limits::max<T>
    MATH_FUNC basic_ray(vector<3, T> const& o, vector<3, T> const& d);

    // Constructor with origin, direction, tmin, and tmax
    MATH_FUNC basic_ray(
            vector<3, T> const& o,
            vector<3, T> const& d,
            T const& tmin,
            T const& tmax
            );

};

} // MATH_NAMESPACE

#include "detail/ray.inl"

#endif // VSNRAY_MATH_RAY_H
