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
inline vector<4, T> tex2D_impl_expand_types(
        T const*                                tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = FloatT;

    T res = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    return { res, T(0.0), T(0.0), T(0.0) };
}

template <
    size_t Dim,
    typename T,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, T> tex2D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, FloatT>;

    auto res = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    vector<4, T> result;

    for (size_t i = 0; i < 4; ++i)
    {
        if (i < Dim)
        {
            result[i] = res[i];
        }
        else
        {
            result[i] = T(0.0);
        }
    }

    return result;
}


// normalized floating point texture, non-simd coordinates

template <
    size_t Dim,
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> tex2D_impl_expand_types(
        vector<Dim, unorm<Bits>> const*         tex,
        vector<2, FloatT> const&                coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<Dim, int>; // TODO: check if this is correct...
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

    for (size_t d = 0; d < 4; ++d)
    {
        if (d < Dim)
        {
            result[d] = unorm_to_float<Bits>(tmp[d]);
        }
        else
        {
            result[d] = FloatT(0.0f);
        }
    }

    return result;
}


// any texture, simd coordinates

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> tex2D_impl_expand_types(
        T const*                                    tex,
        vector<2, FloatT> const&                    coord,
        vector<2, simd::int_type_t<FloatT>> const&  texsize,
        tex_filter_mode                             filter_mode,
        std::array<tex_address_mode, 2> const&      address_mode
        )
{
    using return_type   = FloatT;
    using internal_type = FloatT;

    FloatT res = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    FloatT zero(0.0f);
    return { res, zero, zero, zero };
}

template <
    size_t Dim,
    typename T,
    typename FloatT,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> tex2D_impl_expand_types(
        vector<Dim, T> const*                       tex,
        vector<2, FloatT> const&                    coord,
        vector<2, simd::int_type_t<FloatT>> const&  texsize,
        tex_filter_mode                             filter_mode,
        std::array<tex_address_mode, 2> const&      address_mode
        )
{
    using return_type   = vector<Dim, FloatT>;
    using internal_type = vector<Dim, FloatT>;

    auto res = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    vector<4, FloatT> result;

    for (size_t i = 0; i < 4; ++i)
    {
        if (i < Dim)
        {
            result[i] = res[i];
        }
        else
        {
            result[i] = FloatT(0.0);
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// tex2D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex2D_impl(Tex const& tex, vector<2, FloatT> coord)
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

    return apply_color_conversion(tex2D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            ), tex.get_color_space());
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H
