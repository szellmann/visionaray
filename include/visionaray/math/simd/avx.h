// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_AVX_H
#define VSNRAY_MATH_SIMD_AVH_H 1

#include <visionaray/detail/macros.h>

#include "detail/common.h"
#include "forward.h"
#include "intrinsics.h"

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float8
//

template <>
class basic_float<__m256>
{
public:

    typedef __m256 value_type;
    __m256 value;

    basic_float() = default;
    basic_float(
            float x1,
            float x2,
            float x3,
            float x4,
            float x5,
            float x6,
            float x7,
            float x8
            );
    basic_float(float const v[8]);
    basic_float(float s);
    basic_float(__m256i const& i);
    basic_float(__m256 const& v);

    operator __m256() const;
};


//-------------------------------------------------------------------------------------------------
// int8
//

template <>
class basic_int<__m256i>
{
public:

    typedef __m256i value_type;
    __m256i value;

    basic_int() = default;
    basic_int(
            int x1,
            int x2,
            int x3,
            int x4,
            int x5,
            int x6,
            int x7,
            int x8
            );
    basic_int(int const v[8]);
    basic_int(int s);
    basic_int(unsigned s);
    basic_int(basic_float<__m256> const& f);
    basic_int(__m256i const& v);

    operator __m256i() const;
};


//-------------------------------------------------------------------------------------------------
// mask8
//

template <>
union basic_mask<__m256, __m256i>
{
public:

    __m256  f;
    __m256i i;

    basic_mask() = default;
    basic_mask(__m256 const& m);
    basic_mask(__m256i const& m);
    basic_mask(bool b);
    basic_mask(
            bool x1,
            bool x2,
            bool x3,
            bool x4,
            bool x5,
            bool x6,
            bool x7,
            bool x8
            );
    basic_mask(bool const v[8]);
};

} // simd
} // MATH_NAMESPACE

#include "detail/avx/mask8.inl"
#include "detail/avx/float8.inl"
#include "detail/avx/int8.inl"

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

#endif // VSNRAY_MATH_SIMD_AVX_H
