// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_INTERVAL_H
#define VSNRAY_MATH_INTERVAL_H 1

#include "config.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template <typename T>
class interval
{
public:

    typedef T value_type;

    T min;
    T max;

    interval() = default;
    MATH_FUNC interval(T const& t);
    MATH_FUNC interval(T const& lo, T const& up);

    MATH_FUNC interval& extend(T const &t);
    MATH_FUNC interval& extend(interval<T> const& t);
};

} // MATH_NAMESPACE

#include "detail/interval.inl"

#endif // VSNRAY_MATH_INTERVAL_H
