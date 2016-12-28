// This file is distributed under the MIT license.
// See the LICENSE file for details

#pragma once

#ifndef VSNRAY_MATH_SIMD_AVX512_H
#define VSNRAY_MATH_SIMD_AVX512_H 1

#include <visionaray/detail/macros.h>

#include "detail/common.h"
#include "forward.h"
#include "intrinsics.h"

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float16
//

template <>
class basic_float<__m512>
{
public:

    typedef __m512 value_type;
    __m512 value;

    basic_float() = default;
    basic_float(
            float  x1, float  x2, float  x3, float  x4,
            float  x5, float  x6, float  x7, float  x8,
            float  x9, float x10, float x11, float x12,
            float x13, float x14, float x15, float x16
            );
    basic_float(float const v[16]);
    basic_float(float s);
    basic_float(__m512i const& i);
    basic_float(__m512 const& v);

    operator __m512() const;
};


//-------------------------------------------------------------------------------------------------
// int16
//

template <>
class basic_int<__m512i>
{
public:

    typedef __m512i value_type;
    __m512i value;

    basic_int() = default;
    basic_int(
            int  x1, int  x2, int  x3, int  x4,
            int  x5, int  x6, int  x7, int  x8,
            int  x9, int x10, int x11, int x12,
            int x13, int x14, int x15, int x16
            );
    basic_int(int const v[16]);
    basic_int(int s);
    basic_int(unsigned s);
    basic_int(basic_float<__m512> const& f);
    basic_int(__m512i const& v);

    operator __m512i() const;
};


//-------------------------------------------------------------------------------------------------
// mask16
//

template <>
union basic_mask<__mmask16>
{
public:

    __mmask16 value;

    basic_mask() = default;
    basic_mask(__mmask16 const& m);
    basic_mask(bool b);
    basic_mask(
            bool  x1, bool  x2, bool  x3, bool  x4,
            bool  x5, bool  x6, bool  x7, bool  x8,
            bool  x9, bool x10, bool x11, bool x12,
            bool x13, bool x14, bool x15, bool x16
            );
    basic_mask(bool const v[16]);

    operator __mmask16() const;
};

} // simd
} // MATH_NAMESPACE

#include "detail/avx512/mask16.inl"
#include "detail/avx512/float16.inl"
#include "detail/avx512/int16.inl"

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

#endif // VSNRAY_MATH_SIMD_AVX512_H
