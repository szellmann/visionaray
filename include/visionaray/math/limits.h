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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2)

template <>
class numeric_limits<simd::float4>
{
public:

    MATH_CPU_FUNC static simd::float4 min();
    MATH_CPU_FUNC static simd::float4 lowest();
    MATH_CPU_FUNC static simd::float4 max();
    MATH_CPU_FUNC static simd::float4 epsilon();
};

template <>
class numeric_limits<simd::int4>
{
public:

    MATH_CPU_FUNC static simd::int4 min();
    MATH_CPU_FUNC static simd::int4 lowest();
    MATH_CPU_FUNC static simd::int4 max();
};

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2)

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

template <>
class numeric_limits<simd::float8>
{
public:

    MATH_CPU_FUNC static simd::float8 min();
    MATH_CPU_FUNC static simd::float8 lowest();
    MATH_CPU_FUNC static simd::float8 max();
    MATH_CPU_FUNC static simd::float8 epsilon();
};

template <>
class numeric_limits<simd::int8>
{
public:

    MATH_CPU_FUNC static simd::int8 min();
    MATH_CPU_FUNC static simd::int8 lowest();
    MATH_CPU_FUNC static simd::int8 max();
};

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

template <>
class numeric_limits<simd::float16>
{
public:

    MATH_CPU_FUNC static simd::float16 min();
    MATH_CPU_FUNC static simd::float16 lowest();
    MATH_CPU_FUNC static simd::float16 max();
    MATH_CPU_FUNC static simd::float16 epsilon();
};

template <>
class numeric_limits<simd::int16>
{
public:

    MATH_CPU_FUNC static simd::int16 min();
    MATH_CPU_FUNC static simd::int16 lowest();
    MATH_CPU_FUNC static simd::int16 max();
};

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

} // MATH_NAMESPACE

#include "detail/limits.inl"

#endif // VSNRAY_MATH_LIMITS
