// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H 1

#include <array>
#include <type_traits>
#include <utility>

#include <visionaray/math/detail/math.h>
#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/unorm.h>
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

// normalized floating point texture, non-simd coordinates

template <
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits> const*                      tex,
        vector<3, FloatT> const&                coord,
        vector<3, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
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


// any texture, simd coordinates

// template <
//     typename T,
//     typename FloatT,
//     typename = typename std::enable_if<!std::is_integral<T>::value>::type,
//     typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
//     >
// inline FloatT texND_impl_expand_types(
//         T const*                                    tex,
//         vector<3, FloatT> const&                    coord,
//         vector<3, simd::int_type_t<FloatT>> const&  texsize,
//         tex_filter_mode                             filter_mode,
//         std::array<tex_address_mode, 3> const&      address_mode
//         )
// {
//     using return_type   = FloatT;
//     using internal_type = FloatT;
// 
//     return choose_filter(
//             return_type{},
//             internal_type{},
//             tex,
//             coord,
//             texsize,
//             filter_mode,
//             address_mode
//             );
// }


// normalized floating point texture, simd coordinates

template <
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT texND_impl_expand_types(
        unorm<Bits> const*                          tex,
        vector<3, FloatT> const&                    coord,
        vector<3, simd::int_type_t<FloatT>> const&  texsize,
        tex_filter_mode                             filter_mode,
        std::array<tex_address_mode, 3> const&      address_mode
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
    typename T,
    typename FloatT,
    typename = typename std::enable_if<std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline simd::int_type_t<FloatT> texND_impl_expand_types(
        T const*                                    tex,
        vector<3, FloatT> const&                    coord,
        vector<3, simd::int_type_t<FloatT>> const&  texsize,
        tex_filter_mode                             filter_mode,
        std::array<tex_address_mode, 3> const&      address_mode
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

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H
