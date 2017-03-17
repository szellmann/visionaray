// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H
#define VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H 1

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <visionaray/math/math.h>

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
inline T tex3D_impl_expand_types(
        T const*                                tex,
        vector<3, FloatT> const&                coord,
        vector<3, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
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
inline vector<Dim, T> tex3D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        vector<3, FloatT> const&                coord,
        vector<3, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
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
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<std::is_floating_point<FloatT>::value>::type,
    typename = typename std::enable_if<!simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT tex3D_impl_expand_types(
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


// any texture, non-simd coordinates

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT tex3D_impl_expand_types(
        T const*                                    tex,
        vector<3, FloatT> const&                    coord,
        vector<3, simd::int_type_t<FloatT>> const&  texsize,
        tex_filter_mode                             filter_mode,
        std::array<tex_address_mode, 3> const&      address_mode
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


// normalized floating point texture, simd coordinates

template <
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT tex3D_impl_expand_types(
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
inline simd::int_type_t<FloatT> tex3D_impl_expand_types(
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


//-------------------------------------------------------------------------------------------------
// tex3D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex3D(Tex const& tex, vector<3, FloatT> coord)
    -> decltype( tex3D_impl_expand_types(
            tex.data(),
            coord,
            vector<3, decltype(convert_to_int(std::declval<FloatT>()))>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    using I = simd::int_type_t<FloatT>;

    vector<3, I> texsize(
            static_cast<int>(tex.width()),
            static_cast<int>(tex.height()),
            static_cast<int>(tex.depth())
            );

    return tex3D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_SAMPLER3D_H
