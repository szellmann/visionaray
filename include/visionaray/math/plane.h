// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_PLANE_H
#define VSNRAY_MATH_PLANE_H 1

#include <cstddef>

#include "primitive.h"

namespace MATH_NAMESPACE
{

template <size_t Dim, typename T, typename P>
class basic_plane : public primitive<P>
{
public:

    typedef T value_type;
    typedef vector<Dim, T> vec_type;

    vec_type normal;
    value_type offset;

    basic_plane() = default;

    // NOTE: n must be normalized!
    basic_plane(vec_type const& n, value_type o);
    basic_plane(vec_type const& n, vec_type const& p);

};

} // MATH_NAMESPACE

#include "detail/plane.inl"

#endif // VSNRAY_MATH_PLANE_H
