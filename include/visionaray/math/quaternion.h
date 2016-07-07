// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_QUATERNION_H
#define VSNRAY_MATH_QUATERNION_H 1

#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T>
class quaternion
{
public:

    T w;
    T x;
    T y;
    T z;

    quaternion();
    quaternion(T const& w, T const& x, T const& y, T const& z);
    quaternion(T const& w, vector<3, T> const& v);

    static quaternion identity();
    static quaternion rotation(vector<3, T> const& from, vector<3, T> const& to);

};

} // MATH_NAMESPACE

#include "detail/quaternion.inl"

#endif // VSNRAY_MATH_QUATERNION_H
