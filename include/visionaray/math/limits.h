// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_LIMITS_H
#define VSNRAY_MATH_LIMITS_H 1

#include "simd/simd.h"

#include "norm.h"

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
class numeric_limits<char>
{
public:

    MATH_FUNC static char min();
    MATH_FUNC static char lowest();
    MATH_FUNC static char max();

};

template <>
class numeric_limits<unsigned char>
{
public:

    MATH_FUNC static unsigned char min();
    MATH_FUNC static unsigned char lowest();
    MATH_FUNC static unsigned char max();

};

template <>
class numeric_limits<short>
{
public:

    MATH_FUNC static short min();
    MATH_FUNC static short lowest();
    MATH_FUNC static short max();

};

template <>
class numeric_limits<unsigned short>
{
public:

    MATH_FUNC static unsigned short min();
    MATH_FUNC static unsigned short lowest();
    MATH_FUNC static unsigned short max();

};

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
class numeric_limits<long>
{
public:

    MATH_FUNC static long min();
    MATH_FUNC static long lowest();
    MATH_FUNC static long max();

};

template <>
class numeric_limits<unsigned long>
{
public:

    MATH_FUNC static unsigned long min();
    MATH_FUNC static unsigned long lowest();
    MATH_FUNC static unsigned long max();

};

template <>
class numeric_limits<long long>
{
public:

    MATH_FUNC static long long min();
    MATH_FUNC static long long lowest();
    MATH_FUNC static long long max();

};

template <>
class numeric_limits<unsigned long long>
{
public:

    MATH_FUNC static unsigned long long min();
    MATH_FUNC static unsigned long long lowest();
    MATH_FUNC static unsigned long long max();

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

template <>
class numeric_limits<long double>
{
public:

    MATH_FUNC static long double min();
    MATH_FUNC static long double lowest();
    MATH_FUNC static long double max();
    MATH_FUNC static long double epsilon();

};

template <unsigned Bits>
class numeric_limits<snorm<Bits>>
{
public:

    MATH_FUNC static snorm<Bits> min();
    MATH_FUNC static snorm<Bits> lowest();
    MATH_FUNC static snorm<Bits> max();

};

template <unsigned Bits>
class numeric_limits<unorm<Bits>>
{
public:

    MATH_FUNC static unorm<Bits> min();
    MATH_FUNC static unorm<Bits> lowest();
    MATH_FUNC static unorm<Bits> max();

};


//-------------------------------------------------------------------------------------------------
// SIMD specializations
//

template <typename T>
class numeric_limits<simd::basic_float<T>>
{
public:

    MATH_FUNC static simd::basic_float<T> min();
    MATH_FUNC static simd::basic_float<T> lowest();
    MATH_FUNC static simd::basic_float<T> max();
    MATH_FUNC static simd::basic_float<T> epsilon();
};

template <typename T>
class numeric_limits<simd::basic_int<T>>
{
public:

    MATH_FUNC static simd::basic_int<T> min();
    MATH_FUNC static simd::basic_int<T> lowest();
    MATH_FUNC static simd::basic_int<T> max();
};

} // MATH_NAMESPACE

#include "detail/limits.inl"

#endif // VSNRAY_MATH_LIMITS
