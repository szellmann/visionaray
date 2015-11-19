// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SIMD_TYPE_TRAITS_H
#define VSNRAY_SIMD_TYPE_TRAITS_H 1

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Traits for SIMD vector types
//
//
// Traits have default implementations that are compatible with elementary C++ types
//
//
//  - min_align:
//      get the minimum alignment necessary for data used with SIMD vector type
//      default: value := 16
//
//  - mask_type:
//      get a compatible mask type for a SIMD vector type
//      default: type := bool
//
//  - element_type:
//      get the elementary type of a SIMD vector component
//      default: T <= T
//
//  - num_elements:
//      get the number of vector components for a SIMD vector type
//      default: value := 1
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Minimum alignment, use when allocating data used with SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct min_align
{
    enum { value = 16 };
};

// SSE ----------------------------------------------------

template <>
struct min_align<simd::float4>
{
    enum { value = 16 };
};

template <>
struct min_align<simd::int4>
{
    enum { value = 16 };
};

template <>
struct min_align<simd::mask4>
{
    enum { value = 16 };
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct min_align<simd::float8>
{
    enum { value = 32 };
};

template <>
struct min_align<simd::int8>
{
    enum { value = 32 };
};

template <>
struct min_align<simd::mask8>
{
    enum { value = 32 };
};

#endif


//-------------------------------------------------------------------------------------------------
// Deduce mask type from SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct mask_type
{
    using type = bool;
};

// SSE ----------------------------------------------------

template <>
struct mask_type<simd::float4>
{
    using type = simd::mask4;
};

template <>
struct mask_type<simd::int4>
{
    using type = simd::mask4;
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct mask_type<simd::float8>
{
    using type = simd::mask8;
};

template <>
struct mask_type<simd::int8>
{
    using type = simd::mask8;
};

#endif


//-------------------------------------------------------------------------------------------------
// Get the elementary type belonging to a SIMD vector type
//

// general ------------------------------------------------

template <typename T>
struct element_type
{
    using type = T;
};

// SSE ----------------------------------------------------

template <>
struct element_type<simd::float4>
{
    using type = float;
};

template <>
struct element_type<simd::int4>
{
    using type = int;
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct element_type<simd::float8>
{
    using type = float;
};

template <>
struct element_type<simd::int8>
{
    using type = int;
};

#endif


//-------------------------------------------------------------------------------------------------
// Deduce number of vector components from SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct num_elements
{
    enum { value = 1 };
};

// SSE ----------------------------------------------------

template <>
struct num_elements<simd::float4>
{
    enum { value = 4 };
};

template <>
struct num_elements<simd::int4>
{
    enum { value = 4 };
};

template <>
struct num_elements<simd::mask4>
{
    enum { value = 4 };
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct num_elements<simd::float8>
{
    enum { value = 8 };
};

template <>
struct num_elements<simd::int8>
{
    enum { value = 8 };
};

template <>
struct num_elements<simd::mask8>
{
    enum { value = 8 };
};

#endif

} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_SIMD_TYPE_TRAITS_H
