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

    vec_type ori;
    vec_type dir;

    basic_ray() = default;
    MATH_FUNC basic_ray(vector<3, T> const& o, vector<3, T> const& d);

};

} // MATH_NAMESPACE

#include "detail/ray.inl"

#endif // VSNRAY_MATH_RAY_H
