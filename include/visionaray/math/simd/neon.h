// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_NEON_H
#define VSNRAY_MATH_SIMD_NEON_H 1

#include <visionaray/detail/macros.h>

#include "detail/common.h"
#include "forward.h"
#include "intrinsics.h"

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4
//

template <>
class basic_float<float32x4_t>
{
public:

    typedef float32x4_t value_type;
    float32x4_t value;

    basic_float() = default;
    basic_float(float x, float y, float z, float w);
    basic_float(float const v[4]);
    basic_float(float s);
    basic_float(int32x4_t const& i);
    basic_float(float32x4_t const& v);

    operator float32x4_t() const;
};


//-------------------------------------------------------------------------------------------------
// float8
//

template <>
class basic_float<float32x4_t[2]>
{
public:

    typedef float32x4_t value_type[2];
    float32x4_t value[2];

    basic_float() = default;
    basic_float(float32x4_t const& v1, float32x4_t const& v2);
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
    basic_float(int8 const& i);

    operator float32x4_t*() const;
};


//-------------------------------------------------------------------------------------------------
// int4
//

template <>
class basic_int<int32x4_t>
{
public:

    typedef int32x4_t value_type;
    int32x4_t value;

    basic_int() = default;
    basic_int(int x, int y, int z, int w);
    basic_int(int const v[4]);
    basic_int(int s);
    basic_int(unsigned s);
    basic_int(basic_float<float32x4_t> const& f);
    basic_int(int32x4_t const& v);

    operator int32x4_t() const;
};


//-------------------------------------------------------------------------------------------------
// int8
//

template <>
class basic_int<int32x4_t[2]>
{
public:

    typedef int32x4_t value_type[2];
    int32x4_t value[2];

    basic_int() = default;
    basic_int(int32x4_t const& v1, int32x4_t const& v2);
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
};


//-------------------------------------------------------------------------------------------------
// uint8 (to complement uint32x4_t that is used in some places with NEON)
//

template <>
class basic_uint<uint32x4_t[2]>
{
public:

    typedef uint32x4_t value_type[2];
    uint32x4_t value[2];

    basic_uint() = default;
    basic_uint(uint32x4_t const& v1, uint32x4_t const& v2);
};


//-------------------------------------------------------------------------------------------------
// mask4
//

template <>
union basic_mask<uint32x4_t>
{
public:

    uint32x4_t i;

    basic_mask() = default;
    basic_mask(uint32x4_t const& m);
    basic_mask(int32x4_t const& m);
    basic_mask(bool x, bool y, bool z, bool w);
    basic_mask(bool const v[4]);
    basic_mask(bool b);
};


//-------------------------------------------------------------------------------------------------
// mask8
//

template <>
union basic_mask<uint32x4_t[2]>
{
public:

    uint32x4_t i[2];

    basic_mask() = default;
    basic_mask(uint32x4_t const& i1, uint32x4_t const& i2);
    basic_mask(int8 const& m);
    basic_mask(bool b);
};

} // simd
} // MATH_NAMESPACE

#include "detail/neon/mask4.inl"
#include "detail/neon/mask8.inl"
#include "detail/neon/float4.inl"
#include "detail/neon/float8.inl"
#include "detail/neon/int4.inl"
#include "detail/neon/int8.inl"
#include "detail/neon/uint8.inl"

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

#endif // VSNRAY_MATH_SIMD_NEON_H
