// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_AABB_H
#define VSNRAY_MATH_AABB_H 1

#include <visionaray/detail/macros.h>

#if VSNRAY_CXXLIB_HAS_HDR_ARRAY
#include <array>
#else
#include <boost/array.hpp>
#endif

#include "vector.h"


namespace MATH_NAMESPACE
{


template <typename T>
class basic_aabb
{
public:

    typedef T value_type;
    typedef vector<3, T> vec_type;
#if VSNRAY_CXXLIB_HAS_HDR_ARRAY
    typedef std::array<vec_type, 8> vertex_list;
#else
    typedef boost::array<vec_type, 8> vertex_list;
#endif

    vec_type min;
    vec_type max;

    MATH_FUNC basic_aabb();
    MATH_FUNC basic_aabb(vec_type const& min, vec_type const& max);

    template <typename U>
    MATH_FUNC basic_aabb(basic_aabb<U> const& rhs);

    template <typename U>
    MATH_FUNC basic_aabb(vector<3, U> const& min, vector<3, U> const& max);

    template <typename U>
    MATH_FUNC basic_aabb& operator=(basic_aabb<U> const& rhs);

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
