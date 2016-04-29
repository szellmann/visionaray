// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER3D_H
#define VSNRAY_TEXTURE_SAMPLER3D_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include <visionaray/math/math.h>

#include "sampler_common.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{


template <typename T>
inline T index(T x, T y, T z, vector<3, T> texsize)
{
    return z * texsize[0] * texsize[1] + y * texsize[0] + x;
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
        vector<3, FloatT>                       coord,
        vector<3, SizeT>                        texsize,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    coord = map_tex_coord(coord, texsize, address_mode);

    auto lo = convert_to_int(coord * convert_to_float(texsize));

    auto idx = index(lo[0], lo[1], lo[2], texsize);
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
        vector<3, FloatT>                       coord,
        vector<3, SizeT>                        texsize,
        std::array<tex_address_mode, 3> const&  address_mode
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

    InternalT samples[8] =
    {
        InternalT( point(tex, index( lo.x, lo.y, lo.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, lo.y, lo.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( lo.x, hi.y, lo.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, hi.y, lo.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( lo.x, lo.y, hi.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, lo.y, hi.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( lo.x, hi.y, hi.z, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, hi.y, hi.z, texsize ), ReturnT()) )
    };


    auto uvw = coord1 * convert_to_float(texsize) - convert_to_float(lo);

    auto p1  = lerp(samples[0], samples[1], uvw[0]);
    auto p2  = lerp(samples[2], samples[3], uvw[0]);
    auto p3  = lerp(samples[4], samples[5], uvw[0]);
    auto p4  = lerp(samples[6], samples[7], uvw[0]);

    auto p12 = lerp(p1, p2, uvw[1]);
    auto p34 = lerp(p3, p4, uvw[1]);

    return ReturnT(lerp(p12, p34, uvw[2]));
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT cubic8(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<3, FloatT>                       coord,
        vector<3, SizeT>                        texsize,
        std::array<tex_address_mode, 3> const&  address_mode
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

    auto z = coord.z * convert_to_float(texsize.z) - FloatT(0.5);
    auto floorz = floor(z);
    auto fracz  = z - floor(z);


    auto tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_000  = ( floorx - FloatT(0.5) + tmp000 ) / convert_to_float(texsize.x);

    auto tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_100  = ( floorx + FloatT(1.5) + tmp100 ) / convert_to_float(texsize.x);

    auto tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_010  = ( floory - FloatT(0.5) + tmp010 ) / convert_to_float(texsize.y);

    auto tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_110  = ( floory + FloatT(1.5) + tmp110 ) / convert_to_float(texsize.y);

    auto tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    auto h_001  = ( floorz - FloatT(0.5) + tmp001 ) / convert_to_float(texsize.z);

    auto tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    auto h_101  = ( floorz + FloatT(1.5) + tmp101 ) / convert_to_float(texsize.z);


    auto f_000  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_000, h_010, h_001), texsize, address_mode) );
    auto f_100  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_100, h_010, h_001), texsize, address_mode) );
    auto f_010  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_000, h_110, h_001), texsize, address_mode) );
    auto f_110  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_100, h_110, h_001), texsize, address_mode) );

    auto f_001  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_000, h_010, h_101), texsize, address_mode) );
    auto f_101  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_100, h_010, h_101), texsize, address_mode) );
    auto f_011  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_000, h_110 ,h_101), texsize, address_mode) );
    auto f_111  = InternalT( linear(ReturnT(), InternalT(), tex, vector<3, FloatT>(h_100, h_110, h_101), texsize, address_mode) );

    auto f_00   = g0(fracx) * f_000 + g1(fracx) * f_100;
    auto f_10   = g0(fracx) * f_010 + g1(fracx) * f_110;
    auto f_01   = g0(fracx) * f_001 + g1(fracx) * f_101;
    auto f_11   = g0(fracx) * f_011 + g1(fracx) * f_111;

    auto f_0    = g0(fracy) * f_00 + g1(fracy) * f_10;
    auto f_1    = g0(fracy) * f_01 + g1(fracy) * f_11;

    return ReturnT(g0(fracz) * f_0 + g1(fracz) * f_1);
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
        vector<3, FloatT>                       coord,
        vector<3, SizeT>                        texsize,
        std::array<tex_address_mode, 3> const&  address_mode,
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

    vector<3, decltype(convert_to_int(FloatT{}))> pos[4] =
    {
        convert_to_int(coord1 * convert_to_float(texsize)),
        convert_to_int(coord2 * convert_to_float(texsize)),
        convert_to_int(coord3 * convert_to_float(texsize)),
        convert_to_int(coord4 * convert_to_float(texsize))
    };

    auto uvw = (coord2 * convert_to_float(texsize)) - convert_to_float(pos[1]);

    auto sample = [&](int i, int j, int k) -> InternalT
    {
        return InternalT( point(
                tex,
                index(pos[i].x, pos[j].y, pos[k].z, texsize),
                ReturnT()
                ) );
    };

    auto f00 = w0(uvw.x) * sample(0, 0, 0) + w1(uvw.x) * sample(1, 0, 0) + w2(uvw.x) * sample(2, 0, 0) + w3(uvw.x) * sample(3, 0, 0);
    auto f01 = w0(uvw.x) * sample(0, 1, 0) + w1(uvw.x) * sample(1, 1, 0) + w2(uvw.x) * sample(2, 1, 0) + w3(uvw.x) * sample(3, 1, 0);
    auto f02 = w0(uvw.x) * sample(0, 2, 0) + w1(uvw.x) * sample(1, 2, 0) + w2(uvw.x) * sample(2, 2, 0) + w3(uvw.x) * sample(3, 2, 0);
    auto f03 = w0(uvw.x) * sample(0, 3, 0) + w1(uvw.x) * sample(1, 3, 0) + w2(uvw.x) * sample(2, 3, 0) + w3(uvw.x) * sample(3, 3, 0);

    auto f04 = w0(uvw.x) * sample(0, 0, 1) + w1(uvw.x) * sample(1, 0, 1) + w2(uvw.x) * sample(2, 0, 1) + w3(uvw.x) * sample(3, 0, 1);
    auto f05 = w0(uvw.x) * sample(0, 1, 1) + w1(uvw.x) * sample(1, 1, 1) + w2(uvw.x) * sample(2, 1, 1) + w3(uvw.x) * sample(3, 1, 1);
    auto f06 = w0(uvw.x) * sample(0, 2, 1) + w1(uvw.x) * sample(1, 2, 1) + w2(uvw.x) * sample(2, 2, 1) + w3(uvw.x) * sample(3, 2, 1);
    auto f07 = w0(uvw.x) * sample(0, 3, 1) + w1(uvw.x) * sample(1, 3, 1) + w2(uvw.x) * sample(2, 3, 1) + w3(uvw.x) * sample(3, 3, 1);

    auto f08 = w0(uvw.x) * sample(0, 0, 2) + w1(uvw.x) * sample(1, 0, 2) + w2(uvw.x) * sample(2, 0, 2) + w3(uvw.x) * sample(3, 0, 2);
    auto f09 = w0(uvw.x) * sample(0, 1, 2) + w1(uvw.x) * sample(1, 1, 2) + w2(uvw.x) * sample(2, 1, 2) + w3(uvw.x) * sample(3, 1, 2);
    auto f10 = w0(uvw.x) * sample(0, 2, 2) + w1(uvw.x) * sample(1, 2, 2) + w2(uvw.x) * sample(2, 2, 2) + w3(uvw.x) * sample(3, 2, 2);
    auto f11 = w0(uvw.x) * sample(0, 3, 2) + w1(uvw.x) * sample(1, 3, 2) + w2(uvw.x) * sample(2, 3, 2) + w3(uvw.x) * sample(3, 3, 2);

    auto f12 = w0(uvw.x) * sample(0, 0, 3) + w1(uvw.x) * sample(1, 0, 3) + w2(uvw.x) * sample(2, 0, 3) + w3(uvw.x) * sample(3, 0, 3);
    auto f13 = w0(uvw.x) * sample(0, 1, 3) + w1(uvw.x) * sample(1, 1, 3) + w2(uvw.x) * sample(2, 1, 3) + w3(uvw.x) * sample(3, 1, 3);
    auto f14 = w0(uvw.x) * sample(0, 2, 3) + w1(uvw.x) * sample(1, 2, 3) + w2(uvw.x) * sample(2, 2, 3) + w3(uvw.x) * sample(3, 2, 3);
    auto f15 = w0(uvw.x) * sample(0, 3, 3) + w1(uvw.x) * sample(1, 3, 3) + w2(uvw.x) * sample(2, 3, 3) + w3(uvw.x) * sample(3, 3, 3);

    auto f0  = w0(uvw.y) * f00 + w1(uvw.y) * f01 + w2(uvw.y) * f02 + w3(uvw.y) * f03;
    auto f1  = w0(uvw.y) * f04 + w1(uvw.y) * f05 + w2(uvw.y) * f06 + w3(uvw.y) * f07;
    auto f2  = w0(uvw.y) * f08 + w1(uvw.y) * f09 + w2(uvw.y) * f10 + w3(uvw.y) * f11;
    auto f3  = w0(uvw.y) * f12 + w1(uvw.y) * f13 + w2(uvw.y) * f14 + w3(uvw.y) * f15;

    return ReturnT(w0(uvw.z) * f0 + w1(uvw.z) * f1 + w2(uvw.z) * f2 + w3(uvw.z) * f3);
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
inline ReturnT tex3D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<3, FloatT> const&                coord,
        vector<3, SizeT> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
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
        return cubic8(
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

// float

template <typename T>
inline T tex3D_impl_expand_types(
        T const*                                tex,
        vector<3, float> const&                 coord,
        vector<3, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = float;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

template <size_t Dim, typename T>
inline vector<Dim, T> tex3D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        vector<Dim, float> const&               coord,
        vector<Dim, int> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, float>;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// double

template <typename T>
inline T tex3D_impl_expand_types(
        T const*                                tex,
        vector<3, double> const&                coord,
        vector<3, int> const&                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = double;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

template <size_t Dim, typename T>
inline vector<Dim, T> tex3D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        vector<Dim, double> const&              coord,
        vector<Dim, int> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, double>;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// simd, overload for floating point arrays

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<!std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT tex3D_impl_expand_types(
        T const*                                                tex,
        vector<3, FloatT> const&                                coord,
        vector<3, typename simd::int_type<FloatT>::type> const& texsize,
        tex_filter_mode                                         filter_mode,
        std::array<tex_address_mode, 3> const&                  address_mode
        )
{
    using return_type   = FloatT;
    using internal_type = FloatT;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

// simd, overload for normalized floating point arrays

template <
    unsigned Bits,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline FloatT tex3D_impl_expand_types(
        unorm<Bits> const*                                      tex,
        vector<3, FloatT> const&                                coord,
        vector<3, typename simd::int_type<FloatT>::type> const& texsize,
        tex_filter_mode                                         filter_mode,
        std::array<tex_address_mode, 3> const&                  address_mode
        )
{
    using return_type   = typename simd::int_type<FloatT>::type;
    using internal_type = FloatT;

    // use unnormalized types for internal calculations
    // to avoid the normalization overhead
    auto tmp = tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
            reinterpret_cast<typename best_uint<Bits>::type const*>(tex),
            coord,
            texsize,
            filter_mode,
            address_mode
            );

    // normalize only once upon return
    return unorm_to_float<Bits>(tmp);
}

// simd, overload for integer arrays

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<std::is_integral<T>::value>::type,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline typename simd::int_type<FloatT>::type tex3D_impl_expand_types(
        T const*                                                tex,
        vector<3, FloatT> const&                                coord,
        vector<3, typename simd::int_type<FloatT>::type> const& texsize,
        tex_filter_mode                                         filter_mode,
        std::array<tex_address_mode, 3> const&                  address_mode
        )
{
    using return_type   = typename simd::int_type<FloatT>::type;
    using internal_type = FloatT;

    return tex3D_impl_choose_filter(
            return_type(),
            internal_type(),
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

    using I = typename simd::int_type<FloatT>::type;

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

#endif // VSNRAY_TEXTURE_SAMPLER3D_H
