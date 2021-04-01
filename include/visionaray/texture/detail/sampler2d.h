// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include <visionaray/math/detail/math.h>
#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/vector.h>
#include <visionaray/math/unorm.h>

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
    size_t Dim,
    unsigned Bits,
    typename FloatT,
    typename TexSize,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<Dim, FloatT> texND_impl_expand_types(
        vector<Dim, unorm<Bits>> const*         tex,
        vector<2, FloatT> const&                coord,
        TexSize                                 texsize,
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

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER2D_H
