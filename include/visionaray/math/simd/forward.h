// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_FORWARD_H
#define VSNRAY_MATH_SIMD_FORWARD_H

#include "intrinsics.h"
#include "../config.h"
#include "../forward.h"


namespace MATH_NAMESPACE
{
namespace simd
{

template <typename T>
class basic_float;

template <typename T>
class basic_int;

template <typename... Args>
union basic_mask;


//-------------------------------------------------------------------------------------------------
// typedefs
//

typedef basic_int<__m128i>                      int4;
typedef basic_float<__m128>                     float4;
typedef basic_mask<__m128, __m128i>             mask4;

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
typedef basic_int<__m256i>                      int8;
typedef basic_float<__m256>                     float8;
typedef basic_mask<__m256, __m256i>             mask8;
#endif

typedef basic_ray<float4>                       ray4;

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
typedef basic_ray<float8>                       ray8;
#endif


} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_SIMD_FORWARD_H


