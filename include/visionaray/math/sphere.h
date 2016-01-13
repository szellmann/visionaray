// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SPHERE_H
#define VSNRAY_MATH_SPHERE_H 1

#include "primitive.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T, typename P>
class basic_sphere : public primitive<P>
{
public:

    using scalar_type   = T;
    using vec_type      = vector<3, T>;

public:

    MATH_FUNC basic_sphere() = default;
    MATH_FUNC basic_sphere(vector<3, T> const& c, T r);

    vec_type center;
    scalar_type radius;

};

} // MATH_NAMESPACE

#include "detail/sphere.inl"

#endif // VSNRAY_MATH_SPHERE_H
