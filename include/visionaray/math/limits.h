// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_LIMITS_H
#define VSNRAY_MATH_LIMITS_H 1

namespace MATH_NAMESPACE
{


//-------------------------------------------------------------------------------------------------
// generic limits: no CUDA
//

template <typename T>
class numeric_limits
{
public:

    static T min();
    static T lowest();
    static T max();
    static T epsilon();

};


//-------------------------------------------------------------------------------------------------
// specializations
//

template <>
class numeric_limits<int>
{
public:

    MATH_FUNC static int min();
    MATH_FUNC static int lowest();
    MATH_FUNC static int max();

};

template <>
class numeric_limits<unsigned>
{
public:

    MATH_FUNC static unsigned min();
    MATH_FUNC static unsigned lowest();
    MATH_FUNC static unsigned max();

};

template <>
class numeric_limits<float>
{
public:

    MATH_FUNC static float min();
    MATH_FUNC static float lowest();
    MATH_FUNC static float max();
    MATH_FUNC static float epsilon();

};

template <>
class numeric_limits<double>
{
public:

    MATH_FUNC static double min();
    MATH_FUNC static double lowest();
    MATH_FUNC static double max();
    MATH_FUNC static double epsilon();

};

template <unsigned Bits>
class numeric_limits<unorm<Bits>>
{
public:

    MATH_FUNC static unorm<Bits> min();
    MATH_FUNC static unorm<Bits> lowest();
    MATH_FUNC static unorm<Bits> max();

};

} // MATH_NAMESPACE

#include "detail/limits.inl"

#endif // VSNRAY_MATH_LIMITS
