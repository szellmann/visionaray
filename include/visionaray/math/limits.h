// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_LIMITS_H
#define VSNRAY_MATH_LIMITS_H

namespace MATH_NAMESPACE
{


//-------------------------------------------------------------------------------------------------
// generic limits: no CUDA
//

template <typename T>
class numeric_limits
{
public:

    static constexpr T min();
    static constexpr T lowest();
    static constexpr T max();
    static constexpr T epsilon();

};


//-------------------------------------------------------------------------------------------------
// specializations
//

template <>
class numeric_limits<int>
{
public:

    MATH_FUNC static constexpr int min();
    MATH_FUNC static constexpr int lowest();
    MATH_FUNC static constexpr int max();

};

template <>
class numeric_limits<unsigned>
{
public:

    MATH_FUNC static constexpr unsigned min();
    MATH_FUNC static constexpr unsigned lowest();
    MATH_FUNC static constexpr unsigned max();

};

template <>
class numeric_limits<float>
{
public:

    MATH_FUNC static constexpr float min();
    MATH_FUNC static constexpr float lowest();
    MATH_FUNC static constexpr float max();
    MATH_FUNC static constexpr float epsilon();

};

template <>
class numeric_limits<double>
{
public:

    MATH_FUNC static constexpr double min();
    MATH_FUNC static constexpr double lowest();
    MATH_FUNC static constexpr double max();
    MATH_FUNC static constexpr double epsilon();

};

} // MATH_NAMESPACE

#include "detail/limits.inl"

#endif // VSNRAY_MATH_LIMITS
