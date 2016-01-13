// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_RAY_H
#define VSNRAY_MATH_RAY_H 1

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

    MATH_FUNC basic_ray() = default;
    MATH_FUNC basic_ray(vec_type const& o, vec_type const& d)
        : ori(o)
        , dir(d)
    {
    }

};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_RAY_H
