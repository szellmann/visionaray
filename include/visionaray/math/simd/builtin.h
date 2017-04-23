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
class basic_float<float[4]>
{
public:

    typedef float value_type[4];
    float value[4];

    MATH_FUNC basic_float() = default;
    MATH_FUNC basic_float(float x, float y, float z, float w);
    MATH_FUNC basic_float(float const v[4]);
    MATH_FUNC basic_float(float s);
};


//-------------------------------------------------------------------------------------------------
// int4
//

template <>
class basic_int<int[4]>
{
public:

    typedef int value_type[4];
    int value[4];

    MATH_FUNC basic_int() = default;
    MATH_FUNC basic_int(int x, int y, int z, int w);
    MATH_FUNC basic_int(int const v[4]);
    MATH_FUNC basic_int(int s);
    MATH_FUNC basic_int(unsigned s);
};


//-------------------------------------------------------------------------------------------------
// mask4
//

template <>
union basic_mask<bool[4]>
{
public:

    bool value[4];

    MATH_FUNC basic_mask() = default;
    MATH_FUNC basic_mask(bool x, bool y, bool z, bool w);
    MATH_FUNC basic_mask(bool const v[4]);
    MATH_FUNC basic_mask(bool b);
};

} // simd
} // MATH_NAMESPACE

#include "detail/builtin/mask4.inl"
#include "detail/builtin/float4.inl"
#include "detail/builtin/int4.inl"

#endif // VSNRAY_NO_SIMD_ISA

#endif // VSNRAY_SIMD_BUILTIN_H
