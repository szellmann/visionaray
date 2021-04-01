// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER1D_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER1D_H 1

#include <array>
#include <type_traits>

#include <visionaray/math/simd/avx.h>
#include <visionaray/math/simd/builtin.h>
#include <visionaray/math/simd/neon.h>
#include <visionaray/math/simd/sse.h>
#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/vector.h>

#include "filter/arithmetic_types.h"
#include "filter.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2) || VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

// SIMD: SoA textures

template <
    typename FloatT,
    typename TexSize,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::float4 texND_impl_expand_types(
        simd::float4 const*                     tex,
        vector<1, FloatT> const&                coord,
        TexSize                                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = simd::float4;
    using internal_type = simd::float4;

    return choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2) || VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

template <
    typename FloatT,
    typename TexSize,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::float8 texND_impl_expand_types(
        simd::float8 const*                     tex,
        vector<1, FloatT> const&                coord,
        TexSize                                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = simd::float8;
    using internal_type = simd::float8;

    return choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER1D_H
