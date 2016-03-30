// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_GATHER_H
#define VSNRAY_MATH_SIMD_GATHER_H 1

#include <array>

#include "../vector.h"
#include "avx.h"
#include "sse.h"

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Gather loads from memory to SIMD vectors
//
//
// The gather functions use dedicated hardware instructions when applicable
// and resort to software implementations if no instruction is available.
//
//
// Implemented for the following cases:
//  - base address: unorm<N>,            index type: int4
//  - base address: unorm<N>,            index type: int8
//  - base address: float,               index type: int4
//  - base address: float,               index type: int8
//  - base address: int,                 index type: int4
//  - base address: int,                 index type: int8
//
//  - base address: vector<4, float>,    index type: int4
//  - base address: vector<4, float>,    index type: int8
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Gather float4 from N-bit unorm array, N <= 32
// No dedicated AVX2 instruction!
//

template <unsigned Bits>
inline float4 gather(unorm<Bits> const* base_addr, int4 const& index)
{
    static_assert(Bits <= 32, "Incompatible unorm type");

    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], index);

    return float4(
        static_cast<float>(base_addr[indices[0]]),
        static_cast<float>(base_addr[indices[1]]),
        static_cast<float>(base_addr[indices[2]]),
        static_cast<float>(base_addr[indices[3]])
        );
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather float8 from N-bit unorm array, N <= 32
// No dedicated AVX2 instruction!
//

template <unsigned Bits>
inline float8 gather(unorm<Bits> const* base_addr, int8 const& index)
{
    static_assert(Bits <= 32, "Incompatible unorm type");

    VSNRAY_ALIGN(16) int indices[8];
    store(&indices[0], index);

    return float8(
        static_cast<float>(base_addr[indices[0]]),
        static_cast<float>(base_addr[indices[1]]),
        static_cast<float>(base_addr[indices[2]]),
        static_cast<float>(base_addr[indices[3]]),
        static_cast<float>(base_addr[indices[4]]),
        static_cast<float>(base_addr[indices[5]]),
        static_cast<float>(base_addr[indices[6]]),
        static_cast<float>(base_addr[indices[7]])
        );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Gather float4 from 32-bit float array
//

inline float4 gather(float const* base_addr, int4 const& index)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm_i32gather_ps(base_addr, index, 4);
#else

    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], index);

    return float4(
        base_addr[indices[0]],
        base_addr[indices[1]],
        base_addr[indices[2]],
        base_addr[indices[3]]
        );

#endif
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather float8 from 32-bit float array
//

inline float8 gather(float const* base_addr, int8 const& index)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_i32gather_ps(base_addr, index, 4);
#else

    VSNRAY_ALIGN(32) int indices[8];
    store(&indices[0], index);

    return float8(
        base_addr[indices[0]],
        base_addr[indices[1]],
        base_addr[indices[2]],
        base_addr[indices[3]],
        base_addr[indices[4]],
        base_addr[indices[5]],
        base_addr[indices[6]],
        base_addr[indices[7]]
        );

#endif
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Gather int4 from 32-bit integer array
//

inline int4 gather(int const* base_addr, int4 const& index)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm_i32gather_epi32(base_addr, index, 4);
#else

    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], index);

    return int4(
        base_addr[indices[0]],
        base_addr[indices[1]],
        base_addr[indices[2]],
        base_addr[indices[3]]
        );

#endif
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather int8 from 32-bit integer array
//

inline int8 gather(int const* base_addr, int8 const& index)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2
    return _mm256_i32gather_epi32(base_addr, index, 4);
#else

    VSNRAY_ALIGN(32) int indices[8];
    store(&indices[0], index);

    return int8(
        base_addr[indices[0]],
        base_addr[indices[1]],
        base_addr[indices[2]],
        base_addr[indices[3]],
        base_addr[indices[4]],
        base_addr[indices[5]],
        base_addr[indices[6]],
        base_addr[indices[7]]
        );

#endif
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Gather vector<4, float4> from vector<4, float> array
//

inline vector<4, float4> gather(vector<4, float> const* base_addr, int4 const& index)
{
    //-----------------------------------------------------
    // Optimization for AoS data.
    //
    // Data is gathered w/o a context switch to GP
    // registers by transposing to SoA after memory
    // lookup.
    //

    VSNRAY_ALIGN(16) int indices[4];
    store(&indices[0], index * 4);

    float const* tmp = reinterpret_cast<float const*>(base_addr);

    vector<4, simd::float4> result(
            &tmp[0] + indices[0],
            &tmp[0] + indices[1],
            &tmp[0] + indices[2],
            &tmp[0] + indices[3]
            );

    result = transpose(result);
    return result;
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Gather vector<4, float8> from vector<4, float> array
//

inline vector<4, float8> gather(vector<4, float> const* base_addr, int8 const& index)
{
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX2

    float const* tmp = reinterpret_cast<float const*>(base_addr);

    return vector<4, simd::float8>(
            _mm256_i32gather_ps(tmp, index * 4    , 4),
            _mm256_i32gather_ps(tmp, index * 4 + 1, 4),
            _mm256_i32gather_ps(tmp, index * 4 + 2, 4),
            _mm256_i32gather_ps(tmp, index * 4 + 3, 4)
            );

#else

    VSNRAY_ALIGN(32) int indices[8];
    store(&indices[0], index);

    std::array<vector<4, float>, 8> arr{{
            base_addr[indices[0]],
            base_addr[indices[1]],
            base_addr[indices[2]],
            base_addr[indices[3]],
            base_addr[indices[4]],
            base_addr[indices[5]],
            base_addr[indices[6]],
            base_addr[indices[7]]
            }};


    return simd::pack(arr);

#endif
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_SIMD_GATHER_H
