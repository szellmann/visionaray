// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER1D_H
#define VSNRAY_TEXTURE_SAMPLER1D_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include <visionaray/math/math.h>

#include "filter.h"
#include "sampler_common.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

// any texture, non-simd coordinates

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline T tex1D_impl_expand_types(
        T const*                                tex,
        FloatT                                  coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = FloatT;

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

template <
    size_t Dim,
    typename T,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim, T> tex1D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        FloatT                                  coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, FloatT>;

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


// SIMD: AoS textures

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> tex1D_impl_expand_types(
        vector<4, T> const*                             tex,
        FloatT const&                                   coord,
        typename simd::int_type<FloatT>::type const&    texsize,
        tex_filter_mode                                 filter_mode,
        std::array<tex_address_mode, 1> const&          address_mode
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


// SIMD: SoA textures

inline simd::float4 tex1D_impl_expand_types(
        simd::float4 const*                     tex,
        float                                   coord,
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


//-------------------------------------------------------------------------------------------------
// tex1D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex1D(Tex const& tex, FloatT coord)
    -> decltype( tex1D_impl_expand_types(
            tex.data(),
            coord,
            FloatT(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    using I = typename simd::int_type<FloatT>::type;

    I texsize = static_cast<int>(tex.width());

    return tex1D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_SAMPLER1D_H
