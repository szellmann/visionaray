// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMD_SSE_H
#define VSNRAY_SIMD_SSE_H 1

#include <visionaray/detail/macros.h>

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

    VSNRAY_FORCE_INLINE basic_float() = default;
    VSNRAY_FORCE_INLINE basic_float(float x, float y, float z, float w);
    VSNRAY_FORCE_INLINE basic_float(float const v[4]);
    VSNRAY_FORCE_INLINE basic_float(float s);
    VSNRAY_FORCE_INLINE basic_float(__m128i const& i);
    VSNRAY_FORCE_INLINE basic_float(__m128 const& v);

    VSNRAY_FORCE_INLINE operator __m128() const;
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

    VSNRAY_FORCE_INLINE basic_int() = default;
    VSNRAY_FORCE_INLINE basic_int(int x, int y, int z, int w);
    VSNRAY_FORCE_INLINE basic_int(int const v[4]);
    VSNRAY_FORCE_INLINE basic_int(int s);
    VSNRAY_FORCE_INLINE basic_int(unsigned s);
    VSNRAY_FORCE_INLINE basic_int(basic_float<__m128> const& f);
    VSNRAY_FORCE_INLINE basic_int(__m128i const& v);

    VSNRAY_FORCE_INLINE operator __m128i() const;
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

    VSNRAY_FORCE_INLINE basic_mask() = default;
    VSNRAY_FORCE_INLINE basic_mask(__m128 m);
    VSNRAY_FORCE_INLINE basic_mask(__m128i m);
    VSNRAY_FORCE_INLINE basic_mask(bool x, bool y, bool z, bool w);
    VSNRAY_FORCE_INLINE basic_mask(bool b);
    VSNRAY_FORCE_INLINE basic_mask(basic_float<__m128> const& m);

    VSNRAY_FORCE_INLINE operator basic_float<__m128>() const;
};

} // simd
} // MATH_NAMESPACE

#include "detail/sse/mask4.inl"
#include "detail/sse/float4.inl"
#include "detail/sse/int4.inl"
#include "detail/basic_float.inl"
#include "detail/basic_int.inl"
#include "detail/basic_mask.inl"

#endif // VSNRAY_SIMD_SSE_H
