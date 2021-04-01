// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER_H 1

#include <cstddef>
#include <type_traits>

#include "filter/arithmetic_types.h"
#include "filter.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{

template <typename T>
struct is_unorm : std::false_type {};

template <unsigned Bits>
struct is_unorm<unorm<Bits>> : std::true_type {};

//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

// any texture, non-simd coordinates

template <
    typename T,
    typename Tex,
    typename CoordinateType,
    typename AT = arithmetic_types<T, typename CoordinateType::value_type>
    >
inline typename AT::return_type texND_impl_expand_types(
        T              /* */,
        Tex const&     tex,
        CoordinateType coord
        )
{
    using return_type   = typename AT::return_type;
    using internal_type = typename AT::internal_type;

    return choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord
            );
}

// Overload for unorm textures, non-simd coordinates
template <
    unsigned Bits,
    typename Tex,
    size_t Dims,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits>                 /* */,
        Tex const&                  tex,
        vector<Dims, FloatT> const& coord
        )
{
    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    using return_type   = typename best_uint<Bits>::type;
    using internal_type = FloatT;

    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord
            );

    // normalize only once upon return
    return unorm_to_float<Bits>(tmp);
}

// Overload for vectors of unorms, non-simd coordinates
template <
    size_t Dim1,
    unsigned Bits,
    typename Tex,
    size_t Dim2,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim1, FloatT> texND_impl_expand_types(
        vector<Dim1, unorm<Bits>>   /* */,
        Tex const&                  tex,
        vector<Dim2, FloatT> const& coord
        )
{
    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    using return_type   = vector<Dim1, typename best_uint<Bits>::type>;
    using internal_type = vector<Dim1, FloatT>;

    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord
            );

    // normalize only once upon return
    vector<Dim1, FloatT> result;

    for (size_t d = 0; d < Dim1; ++d)
    {
        result[d] = unorm_to_float<Bits>((int)tmp[d]);
    }

    return result;
}

// normalized floating point texture, simd coordinates
template <
    unsigned Bits,
    typename Tex,
    size_t Dim,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits>                /* */,
        Tex const&                 tex,
        vector<Dim, FloatT> const& coord
        )
{
    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    using return_type   = simd::int_type_t<FloatT>;
    using internal_type = FloatT;

    auto tmp = choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord
            );

    // normalize only once upon return
    return unorm_to_float<Bits>(tmp);
}

// integer texture, simd coordinates

template <
    typename T,
    typename Tex,
    size_t Dim,
    typename FloatT,
    typename = typename std::enable_if<std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::int_type_t<FloatT> texND_impl_expand_types(
        T                          /* */,
        Tex const&                 tex,
        vector<Dim, FloatT> const& coord
        )
{
    using return_type   = simd::int_type_t<FloatT>;
    using internal_type = FloatT;

    return choose_filter(
            return_type{},
            internal_type{},
            tex,
            coord
            );
}


//-------------------------------------------------------------------------------------------------
// texND() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex_fetch_impl(Tex const& tex, vector<Tex::dimensions, FloatT> coord)
    -> decltype(texND_impl_expand_types(typename Tex::value_type{}, tex, coord))
{
    return apply_color_conversion(
            texND_impl_expand_types(typename Tex::value_type{}, tex, coord),
            tex.get_color_space()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER_H
