// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_CYLINDER_H
#define VSNRAY_MATH_CYLINDER_H 1

#include "config.h"
#include "primitive.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T, typename P>
class basic_cylinder : public primitive<P>
{
public:

    using scalar_type   = T;
    using vec_type      = vector<3, T>;

public:

    basic_cylinder() = default;
    MATH_FUNC basic_cylinder(vector<3, T> const& v1, vector<3, T> const& v2, T const& r);

    vec_type v1;
    vec_type v2;
    scalar_type radius;

};

} // MATH_NAMESPACE

#include "detail/cylinder.inl"

#endif // VSNRAY_MATH_CYLINDER_H
