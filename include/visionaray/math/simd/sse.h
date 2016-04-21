// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMD_SSE_H
#define VSNRAY_SIMD_SSE_H 1

#include <visionaray/detail/macros.h>

#include "detail/common.h"
#include "forward.h"
#include "intrinsics.h"


namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4
//

template <>
class basic_float<__m128>
{
public:

    typedef __m128 value_type;
    __m128 value;

    basic_float() = default;
    basic_float(float x, float y, float z, float w);
    basic_float(float const v[4]);
    basic_float(float s);
    basic_float(__m128i const& i);
    basic_float(__m128 const& v);

    operator __m128() const;
};


//-------------------------------------------------------------------------------------------------
// int4
//

template <>
class basic_int<__m128i>
{
public:

    typedef __m128i value_type;
    __m128i value;

    basic_int() = default;
    basic_int(int x, int y, int z, int w);
    basic_int(int const v[4]);
    basic_int(int s);
    basic_int(unsigned s);
    basic_int(basic_float<__m128> const& f);
    basic_int(__m128i const& v);

    operator __m128i() const;
};


//-------------------------------------------------------------------------------------------------
// mask4
//

template <>
union basic_mask<__m128, __m128i>
{
public:

    __m128  f;
    __m128i i;

    basic_mask() = default;
    basic_mask(__m128 m);
    basic_mask(__m128i m);
    basic_mask(bool x, bool y, bool z, bool w);
    basic_mask(bool const v[4]);
    basic_mask(bool b);
    basic_mask(basic_float<__m128> const& m);

    operator basic_float<__m128>() const;
};

} // simd
} // MATH_NAMESPACE

#include "detail/sse/mask4.inl"
#include "detail/sse/float4.inl"
#include "detail/sse/int4.inl"

#endif // VSNRAY_SIMD_SSE_H
