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

// any texture, simd coordinates

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> texND_impl_expand_types(
        vector<4, T> const*                     tex,
        FloatT const&                           coord,
        simd::int_type_t<FloatT> const&         texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = vector<4, FloatT>;
    using internal_type = vector<4, FloatT>;

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


#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2) || VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON_FP)

// SIMD: SoA textures

template <
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::float4 texND_impl_expand_types(
        simd::float4 const*                     tex,
        FloatT                                  coord,
        int                                     texsize,
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
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::float8 texND_impl_expand_types(
        simd::float8 const*                     tex,
        FloatT                                  coord,
        int                                     texsize,
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


//-------------------------------------------------------------------------------------------------
// tex1D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex1D_impl(Tex const& tex, FloatT coord)
    -> decltype( texND_impl_expand_types(
            tex.data(),
            coord,
            simd::int_type_t<FloatT>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    using I = simd::int_type_t<FloatT>;

    I texsize = static_cast<int>(tex.width());

    return texND_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER1D_H
