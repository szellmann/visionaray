// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMD_BUILTIN_H
#define VSNRAY_SIMD_BUILTIN_H 1

#include <visionaray/detail/macros.h>

#include "detail/common.h"
#include "forward.h"
#include "intrinsics.h"

#if VSNRAY_NO_SIMD_ISA

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float4
//

template <>
class VSNRAY_ALIGN(16) basic_float<float[4]>
{
public:

    typedef float value_type[4];
    float value[4];

    basic_float() = default;
    MATH_FUNC basic_float(float x, float y, float z, float w);
    MATH_FUNC basic_float(float const v[4]);
    MATH_FUNC basic_float(float s);
    MATH_FUNC basic_float(basic_int<int[4]> const i);
};


//-------------------------------------------------------------------------------------------------
// int4
//

template <>
class VSNRAY_ALIGN(16) basic_int<int[4]>
{
public:

    typedef int value_type[4];
    int value[4];

    basic_int() = default;
    MATH_FUNC basic_int(int x, int y, int z, int w);
    MATH_FUNC basic_int(int const v[4]);
    MATH_FUNC basic_int(int s);
    MATH_FUNC basic_int(unsigned s);
};


//-------------------------------------------------------------------------------------------------
// mask4
//

template <>
union VSNRAY_ALIGN(16) basic_mask<bool[4]>
{
public:

    bool value[4];

    basic_mask() = default;
    MATH_FUNC basic_mask(bool x, bool y, bool z, bool w);
    MATH_FUNC basic_mask(bool const v[4]);
    MATH_FUNC basic_mask(bool b);
    MATH_FUNC basic_mask(basic_int<int[4]> const i);
};

} // simd
} // MATH_NAMESPACE

#include "detail/builtin/mask4.inl"
#include "detail/builtin/float4.inl"
#include "detail/builtin/int4.inl"

#endif // VSNRAY_NO_SIMD_ISA

#if !VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float8
//

template <>
class VSNRAY_ALIGN(32) basic_float<float[8]>
{
public:

    typedef float value_type[8];
    float value[8];

    basic_float() = default;
    MATH_FUNC basic_float(
            float x1,
            float x2,
            float x3,
            float x4,
            float x5,
            float x6,
            float x7,
            float x8
            );
    MATH_FUNC basic_float(float const v[8]);
    MATH_FUNC basic_float(float s);
};


//-------------------------------------------------------------------------------------------------
// int8
//

template <>
class VSNRAY_ALIGN(32) basic_int<int[8]>
{
public:

    typedef int value_type[8];
    int value[8];

    basic_int() = default;
    MATH_FUNC basic_int(
            int x1,
            int x2,
            int x3,
            int x4,
            int x5,
            int x6,
            int x7,
            int x8
            );
    MATH_FUNC basic_int(int const v[8]);
    MATH_FUNC basic_int(int s);
    MATH_FUNC basic_int(unsigned s);
};


//-------------------------------------------------------------------------------------------------
// mask8
//

template <>
union VSNRAY_ALIGN(32) basic_mask<bool[8]>
{
public:

    bool value[8];

    basic_mask() = default;
    MATH_FUNC basic_mask(
            bool x1,
            bool x2,
            bool x3,
            bool x4,
            bool x5,
            bool x6,
            bool x7,
            bool x8
            );
    MATH_FUNC basic_mask(bool const v[8]);
    MATH_FUNC basic_mask(bool b);
};

} // simd
} // MATH_NAMESPACE

#include "detail/builtin/mask8.inl"
#include "detail/builtin/float8.inl"
#include "detail/builtin/int8.inl"

#endif // !VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

#if !VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// float16
//

template <>
class VSNRAY_ALIGN(64) basic_float<float[16]>
{
public:

    typedef float value_type[16];
    float value[16];

    basic_float() = default;
    MATH_FUNC basic_float(
            float  x1, float  x2, float  x3, float  x4,
            float  x5, float  x6, float  x7, float  x8,
            float  x9, float x10, float x11, float x12,
            float x13, float x14, float x15, float x16
            );
    MATH_FUNC basic_float(float const v[16]);
    MATH_FUNC basic_float(float s);
};


//-------------------------------------------------------------------------------------------------
// int16
//

template <>
class VSNRAY_ALIGN(64) basic_int<int[16]>
{
public:

    typedef int value_type[16];
    int value[16];

    basic_int() = default;
    MATH_FUNC basic_int(
            int  x1, int  x2, int  x3, int  x4,
            int  x5, int  x6, int  x7, int  x8,
            int  x9, int x10, int x11, int x12,
            int x13, int x14, int x15, int x16
            );
    MATH_FUNC basic_int(int const v[16]);
    MATH_FUNC basic_int(int s);
    MATH_FUNC basic_int(unsigned s);
};


//-------------------------------------------------------------------------------------------------
// mask16
//

template <>
union VSNRAY_ALIGN(64) basic_mask<bool[16]>
{
public:

    bool value[16];

    basic_mask() = default;
    MATH_FUNC basic_mask(
            bool  x1, bool  x2, bool  x3, bool  x4,
            bool  x5, bool  x6, bool  x7, bool  x8,
            bool  x9, bool x10, bool x11, bool x12,
            bool x13, bool x14, bool x15, bool x16
            );
    MATH_FUNC basic_mask(bool const v[16]);
    MATH_FUNC basic_mask(bool b);
};

} // simd
} // MATH_NAMESPACE

#include "detail/builtin/mask16.inl"
#include "detail/builtin/float16.inl"
#include "detail/builtin/int16.inl"

#endif // !VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

#endif // VSNRAY_SIMD_BUILTIN_H
