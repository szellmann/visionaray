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
//  - alignment_of:
//      get the required alignment for the SIMD vector type
//      default: value := alignof(T)
//
//  - aligned_array:
//      get an array that adheres to the alignment requirements and that can store
//      the contents of a SIMD vector type
//      default: n/a
//
//  - float_type:
//      get a compatible float type for a SIMD vector type
//      default: float
//
//  - int_type
//      get a compatible int type for a SIMD vector type
//      default: int
//
//  - mask_type:
//      get a compatible mask type for a SIMD vector type
//      default: type := bool
//
//  - is_simd_vector
//      check if T is a SIMD vector type
//      default: type := false
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
// Required alignment, use when allocating data used with SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct alignment_of
{
    enum { value = alignof(T) };
};

// SSE ----------------------------------------------------

template <>
struct alignment_of<simd::float4>
{
    enum { value = 16 };
};

template <>
struct alignment_of<simd::int4>
{
    enum { value = 16 };
};

template <>
struct alignment_of<simd::mask4>
{
    enum { value = 16 };
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct alignment_of<simd::float8>
{
    enum { value = 32 };
};

template <>
struct alignment_of<simd::int8>
{
    enum { value = 32 };
};

template <>
struct alignment_of<simd::mask8>
{
    enum { value = 32 };
};

#endif


//-------------------------------------------------------------------------------------------------
// Get an array that adheres to the alignment requirements and that can store
// the contents of a SIMD vector type
//

// general (n/a) ------------------------------------------

template <typename T>
struct aligned_array;

// SSE ----------------------------------------------------

template <>
struct aligned_array<simd::float4>
{
    typedef VSNRAY_ALIGN(16) float type[4];
};

template <>
struct aligned_array<simd::int4>
{
    typedef VSNRAY_ALIGN(16) int type[4];
};

template <>
struct aligned_array<simd::mask4>
{
    typedef VSNRAY_ALIGN(16) int type[4];
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct aligned_array<simd::float8>
{
    typedef VSNRAY_ALIGN(32) float type[8];
};

template <>
struct aligned_array<simd::int8>
{
    typedef VSNRAY_ALIGN(32) int type[8];
};

template <>
struct aligned_array<simd::mask8>
{
    typedef VSNRAY_ALIGN(32) int type[8];
};

#endif


//-------------------------------------------------------------------------------------------------
// Deduce float type from SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct float_type
{
    using type = float;
};

// SSE ----------------------------------------------------

template <>
struct float_type<simd::int4>
{
    using type = simd::float4;
};

template <>
struct float_type<simd::mask4>
{
    using type = simd::float4;
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct float_type<simd::int8>
{
    using type = simd::float8;
};

template <>
struct float_type<simd::mask8>
{
    using type = simd::float8;
};

#endif


//-------------------------------------------------------------------------------------------------
// Deduce int type from SIMD vector types
//

// general ------------------------------------------------

template <typename T>
struct int_type
{
    using type = int;
};

// SSE ----------------------------------------------------

template <>
struct int_type<simd::float4>
{
    using type = simd::int4;
};

template <>
struct int_type<simd::mask4>
{
    using type = simd::int4;
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct int_type<simd::float8>
{
    using type = simd::int8;
};

template <>
struct int_type<simd::mask8>
{
    using type = simd::int8;
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
// Check if a given type T is a SIMD vector type
//

// general ------------------------------------------------

template <typename T>
struct is_simd_vector
{
    enum { value = false };
};

// SSE ----------------------------------------------------

template <>
struct is_simd_vector<simd::float4>
{
    enum { value = true };
};

template <>
struct is_simd_vector<simd::int4>
{
    enum { value = true };
};

template <>
struct is_simd_vector<simd::mask4>
{
    enum { value = true };
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

// AVX ----------------------------------------------------

template <>
struct is_simd_vector<simd::float8>
{
    enum { value = true };
};

template <>
struct is_simd_vector<simd::int8>
{
    enum { value = true };
};

template <>
struct is_simd_vector<simd::mask8>
{
    enum { value = true };
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
