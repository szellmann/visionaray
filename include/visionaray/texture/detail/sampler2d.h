// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H 1

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/detail/math.h>
#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/vector.h>
#include <visionaray/math/unorm.h>

#include "filter.h"
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
inline T tex2D_impl_expand_types(
        T const*                                tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
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
inline vector<Dim, T> tex2D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
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


// normalized floating point texture, non-simd coordinates

template <
    size_t Dim,
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim, FloatT> tex2D_impl_expand_types(
        vector<Dim, unorm<Bits>> const*         tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<Dim, int>;
    using internal_type = vector<Dim, FloatT>;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            reinterpret_cast<vector<Dim, typename best_uint<Bits>::type> const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    vector<Dim, FloatT> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = unorm_to_float<Bits>(tmp[d]);
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// tex2D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex2D(Tex const& tex, vector<2, FloatT> coord)
    -> decltype( tex2D_impl_expand_types(
            tex.data(),
            coord,
            vector<2, decltype(convert_to_int(std::declval<FloatT>()))>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    using I = simd::int_type_t<FloatT>;

    vector<2, I> texsize(
            static_cast<int>(tex.width()),
            static_cast<int>(tex.height())
            );

    return tex2D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H
