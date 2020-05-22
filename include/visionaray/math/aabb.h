// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_AABB_H
#define VSNRAY_MATH_AABB_H 1

#include <cstddef>

#include "config.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T, size_t Dim>
class basic_aabb
{
public:

    typedef T value_type;
    typedef vector<Dim, T> vec_type;

    vec_type min;
    vec_type max;

    basic_aabb() = default;
    MATH_FUNC basic_aabb(vec_type const& min, vec_type const& max);

    template <typename U>
    MATH_FUNC basic_aabb(basic_aabb<U, Dim> const& rhs);

    template <typename U>
    MATH_FUNC basic_aabb(vector<Dim, U> const& min, vector<Dim, U> const& max);

    template <typename U>
    MATH_FUNC basic_aabb& operator=(basic_aabb<U, Dim> const& rhs);

    MATH_FUNC vec_type center() const;
    MATH_FUNC vec_type size() const;
    MATH_FUNC vec_type safe_size() const;

    MATH_FUNC void invalidate();

    MATH_FUNC bool invalid() const;
    MATH_FUNC bool valid() const;

    MATH_FUNC bool empty() const;

    MATH_FUNC bool contains(vec_type const& v) const;
    MATH_FUNC bool contains(basic_aabb const& b) const;

    MATH_FUNC void insert(vec_type const& v);
    MATH_FUNC void insert(basic_aabb const& b);

};

} // MATH_NAMESPACE

#include "detail/aabb.inl"

#endif // VSNRAY_MATH_AABB_H
