// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER2D_H
#define VSNRAY_TEXTURE_SAMPLER2D_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include "sampler_common.h"

namespace visionaray
{
namespace detail
{


template <typename T>
inline T index(T x, T y, vector<2, T> texsize)
{
    return y * texsize[0] + x;
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT nearest(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, SizeT>                        texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    coord = map_tex_coord(coord, texsize, address_mode);

    auto lo = convert_to_int(coord * convert_to_float(texsize));

    auto idx = index(lo[0], lo[1], texsize);
    return point(tex, idx, ReturnT());
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT linear(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, SizeT>                        texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    auto coord1 = map_tex_coord(
            coord - FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord2 = map_tex_coord(
            coord + FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto lo = convert_to_int(coord1 * convert_to_float(texsize));
    auto hi = convert_to_int(coord2 * convert_to_float(texsize));

    InternalT samples[4] =
    {
        InternalT( point(tex, index( lo.x, lo.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, lo.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( lo.x, hi.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, hi.y, texsize ), ReturnT()) )
    };


    auto uv = coord1 * convert_to_float(texsize) - convert_to_float(lo);

    auto p1 = lerp(samples[0], samples[1], uv[0]);
    auto p2 = lerp(samples[2], samples[3], uv[0]);

    return ReturnT(lerp(p1, p2, uv[1]));
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT cubic4(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, SizeT>                        texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;

    auto x = coord.x * convert_to_float(texsize.x) - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * convert_to_float(texsize.y) - FloatT(0.5);
    auto floory = floor(y);
    auto fracy  = y - floor(y);


    auto tmp00 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_00  = ( floorx - FloatT(0.5) + tmp00 ) / texsize.x;

    auto tmp10 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_10  = ( floorx + FloatT(1.5) + tmp10 ) / texsize.x;

    auto tmp01 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_01  = ( floory - FloatT(0.5) + tmp01 ) / texsize.y;

    auto tmp11 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_11  = ( floory + FloatT(1.5) + tmp11 ) / texsize.y;


    auto f_00  = InternalT( linear(ReturnT(), InternalT(), tex, vector<2, FloatT>(h_00, h_01), texsize, address_mode) );
    auto f_10  = InternalT( linear(ReturnT(), InternalT(), tex, vector<2, FloatT>(h_10, h_01), texsize, address_mode) );
    auto f_01  = InternalT( linear(ReturnT(), InternalT(), tex, vector<2, FloatT>(h_00, h_11), texsize, address_mode) );
    auto f_11  = InternalT( linear(ReturnT(), InternalT(), tex, vector<2, FloatT>(h_10, h_11), texsize, address_mode) );

    auto f_0   = g0(fracx) * f_00 + g1(fracx) * f_10;
    auto f_1   = g0(fracx) * f_01 + g1(fracx) * f_11;

    return ReturnT(g0(fracy) * f_0 + g1(fracy) * f_1);
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT,
    typename W0,
    typename W1,
    typename W2,
    typename W3
    >
inline ReturnT cubic(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, SizeT>                        texsize,
        std::array<tex_address_mode, 2> const&  address_mode,
        W0                                      w0,
        W1                                      w1,
        W2                                      w2,
        W3                                      w3
        )
{
    auto coord1 = map_tex_coord(
            coord - FloatT(1.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord2 = map_tex_coord(
            coord - FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord3 = map_tex_coord(
            coord + FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord4 = map_tex_coord(
            coord + FloatT(1.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    vector<2, decltype(convert_to_int(FloatT{}))> pos[4] =
    {
        convert_to_int(coord1 * convert_to_float(texsize)),
        convert_to_int(coord2 * convert_to_float(texsize)),
        convert_to_int(coord3 * convert_to_float(texsize)),
        convert_to_int(coord4 * convert_to_float(texsize))
    };

    auto uv = (coord2 * convert_to_float(texsize)) - convert_to_float(pos[1]);

    auto sample = [&](int i, int j) -> InternalT
    {
        return InternalT( point(
                tex,
                index(pos[i].x, pos[j].y, texsize),
                ReturnT()
                ) );
    };

    auto f0 = w0(uv.x) * sample(0, 0) + w1(uv.x) * sample(1, 0) + w2(uv.x) * sample(2, 0) + w3(uv.x) * sample(3, 0);
    auto f1 = w0(uv.x) * sample(0, 1) + w1(uv.x) * sample(1, 1) + w2(uv.x) * sample(2, 1) + w3(uv.x) * sample(3, 1);
    auto f2 = w0(uv.x) * sample(0, 2) + w1(uv.x) * sample(1, 2) + w2(uv.x) * sample(2, 2) + w3(uv.x) * sample(3, 2);
    auto f3 = w0(uv.x) * sample(0, 3) + w1(uv.x) * sample(1, 3) + w2(uv.x) * sample(2, 3) + w3(uv.x) * sample(3, 3);

    return ReturnT(w0(uv.y) * f0 + w1(uv.y) * f1 + w2(uv.y) * f2 + w3(uv.y) * f3);
}


//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT tex2D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT> const&                coord,
        vector<2, SizeT> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    switch (filter_mode)
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::Linear:
        return linear(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::BSpline:
        return cubic4(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::CardinalSpline:
        return cubic(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode,
                cspline::w0_func<FloatT>(),
                cspline::w1_func<FloatT>(),
                cspline::w2_func<FloatT>(),
                cspline::w3_func<FloatT>()
                );

    }
}


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

    return tex2D_impl_choose_filter(
            return_type(),
            internal_type(),
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

    return tex2D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// normalized floating point texture, non-simd coordinates

template <size_t Dim, unsigned Bits>
inline vector<Dim, float> tex2D_impl_expand_types(
        vector<Dim, unorm<Bits>> const*         tex,
        vector<2, float> const&                 coord,
        vector<2, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<Dim, int>;
    using internal_type = vector<Dim, float>;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = tex2D_impl_choose_filter(
            return_type(),
            internal_type(),
            reinterpret_cast<vector<Dim, typename best_uint<Bits>::type> const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    vector<Dim, float> result;

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

    using I = typename simd::int_type<FloatT>::type;

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

#endif // VSNRAY_TEXTURE_SAMPLER2D_H
