// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER_H 1

#include <type_traits>

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

// any texture, non-simd coordinates

template <
    typename TexelType,
    typename CoordinateType,
    typename TexSize,
    typename  AddressMode,
    typename = typename std::enable_if<std::is_floating_point<CoordinateType>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<CoordinateType>::value>::type,
    typename AT = arithmetic_types<TexelType, CoordinateType>
    >
inline typename AT::return_type texND_impl_expand_types(
        TexelType const* tex,
        CoordinateType   coord,
        TexSize          texsize,
        tex_filter_mode  filter_mode,
        AddressMode      address_mode
        )
{
    using return_type   = typename AT::return_type;
    using internal_type = typename AT::internal_type;

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


// any texture, simd coordinates

template <
    size_t Dim,
    typename T,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        T const*                   tex,
        vector<Dim, FloatT> const& coord,
        TexSize                    texsize,
        tex_filter_mode            filter_mode,
        AddressMode                address_mode
        )
{
    using return_type   = FloatT;
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

// Overload for vector textures, we can prob. get rid of this (TODO!)
template <
    size_t Dim1, // TODO: that's what arithmetic types are for..
    size_t Dim2,
    typename T,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim1, FloatT> texND_impl_expand_types(
        vector<Dim1, T> const*      tex,
        vector<Dim2, FloatT> const& coord,
        TexSize                     texsize,
        tex_filter_mode             filter_mode,
        AddressMode                 address_mode
        )
{
    using return_type   = vector<Dim1, FloatT>;
    using internal_type = vector<Dim1, FloatT>;

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

// Overload for unorm textures, non-simd coordinates
template <
    size_t Dims,
    unsigned Bits,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits> const*          tex,
        vector<Dims, FloatT> const& coord,
        TexSize                     texsize,
        tex_filter_mode             filter_mode,
        AddressMode                 address_mode
        )
{
    using return_type   = int;
    using internal_type = FloatT;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            reinterpret_cast<typename best_uint<Bits>::type const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    return unorm_to_float<Bits>(tmp);
}

// Overload for vectors of unorms, non-simd coordinates
template <
    size_t Dim1,
    size_t Dim2,
    unsigned Bits,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim1, FloatT> texND_impl_expand_types(
        vector<Dim1, unorm<Bits>> const* tex,
        vector<Dim2, FloatT> const&      coord,
        TexSize                          texsize,
        tex_filter_mode                  filter_mode,
        AddressMode                      address_mode
        )
{
    using return_type   = vector<Dim1, int>;
    using internal_type = vector<Dim1, FloatT>;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            reinterpret_cast<vector<Dim1, typename best_uint<Bits>::type> const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    vector<Dim1, FloatT> result;

    for (size_t d = 0; d < Dim1; ++d)
    {
        result[d] = unorm_to_float<Bits>(tmp[d]);
    }

    return result;
}

// normalized floating point texture, simd coordinates
template <
    size_t Dim,
    unsigned Bits,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits> const*         tex,
        vector<Dim, FloatT> const& coord,
        TexSize                    texsize,
        tex_filter_mode            filter_mode,
        AddressMode                address_mode
        )
{
    using return_type   = simd::int_type_t<FloatT>;
    using internal_type = FloatT;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            reinterpret_cast<typename best_uint<Bits>::type const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    return unorm_to_float<Bits>(tmp);
}


// integer texture, simd coordinates

template <
    size_t Dim,
    typename T,
    typename FloatT,
    typename TexSize,
    typename AddressMode,
    typename = typename std::enable_if<std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::int_type_t<FloatT> texND_impl_expand_types(
        T const*                   tex,
        vector<Dim, FloatT> const& coord,
        TexSize                    texsize,
        tex_filter_mode            filter_mode,
        AddressMode                address_mode
        )
{
    using return_type   = simd::int_type_t<FloatT>;
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


//-------------------------------------------------------------------------------------------------
// texND() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex_fetch_impl(Tex const& tex, vector<Tex::dimensions, FloatT> coord)
    -> decltype( texND_impl_expand_types(
            tex.data(),
            coord,
            vector<Tex::dimensions, decltype(convert_to_int(std::declval<FloatT>()))>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    using I = simd::int_type_t<FloatT>;

    vector<Tex::dimensions, I> texsize;
    for (int i = 0; i < Tex::dimensions; ++i)
    {
        texsize[i] = I(static_cast<int>(tex.size()[i]));
    }

    return apply_color_conversion(texND_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            ), tex.get_color_space());
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER_H
