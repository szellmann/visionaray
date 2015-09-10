// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER2D_H
#define VSNRAY_TEXTURE_SAMPLER2D_H 1

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


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT nearest(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, FloatT>                       texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    vector<2, FloatT> lo(
            floor(coord.x * texsize.x),
            floor(coord.y * texsize.y)
            );

    lo[0] = clamp(lo[0], FloatT(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], FloatT(0.0f), texsize[1] - 1);

    FloatT idx = index(lo[0], lo[1], texsize);
    return point(tex, idx, ReturnT());

}


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT linear(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, FloatT>                       texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    vector<2, FloatT> texcoordf( coord * texsize - FloatT(0.5) );

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsize[1] - 1 );

    vector<2, FloatT> lo( floor(texcoordf[0]), floor(texcoordf[1]) );
    vector<2, FloatT> hi( ceil(texcoordf[0]),  ceil(texcoordf[1]) );


    InternalT samples[4] =
    {
        InternalT( point(tex, index( lo.x, lo.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, lo.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( lo.x, hi.y, texsize ), ReturnT()) ),
        InternalT( point(tex, index( hi.x, hi.y, texsize ), ReturnT()) )
    };


    auto uv = texcoordf - lo;

    auto p1 = lerp(samples[0], samples[1], uv[0]);
    auto p2 = lerp(samples[2], samples[3], uv[0]);

    return ReturnT(lerp(p1, p2, uv[1]));
}


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT cubic4(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT>                       coord,
        vector<2, FloatT>                       texsize,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;

    auto x = coord.x * texsize.x - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * texsize.y - FloatT(0.5);
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


//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <typename ReturnT, typename FloatT, typename TexelT, typename InternalT>
inline ReturnT tex2D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<2, FloatT> const&                coord,
        vector<2, FloatT> const&                texsize,
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

    }
}


//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

// float

template <typename T>
inline T tex2D_impl_expand_types(
        T const*                                tex,
        vector<2, float> const&                 coord,
        vector<2, float> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = float;

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

template <typename T>
inline vector<3, T> tex2D_impl_expand_types(
        vector<3, T> const*                     tex,
        vector<2, float> const&                 coord,
        vector<2, float> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<3, T>;
    using internal_type = vector<3, float>;

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

template <typename T>
inline vector<4, T> tex2D_impl_expand_types(
        vector<4, T> const*                     tex,
        vector<2, float> const&                 coord,
        vector<2, float> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<4, T>;
    using internal_type = vector<4, float>;

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


// double

template <typename T>
inline T tex2D_impl_expand_types(
        T const*                                tex,
        vector<2, double> const&                coord,
        vector<2, double> const&                texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = double;

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

template <typename T>
inline vector<3, T> tex2D_impl_expand_types(
        vector<3, T> const*                     tex,
        vector<2, double> const&                coord,
        vector<2, double> const&                texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<3, T>;
    using internal_type = vector<3, double>;

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

template <typename T>
inline vector<4, T> tex2D_impl_expand_types(
        vector<4, T> const*                     tex,
        vector<2, double> const&                coord,
        vector<2, double> const&                texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 2> const&  address_mode
        )
{
    using return_type   = vector<4, T>;
    using internal_type = vector<4, double>;

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


//-------------------------------------------------------------------------------------------------
// tex2D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex2D(Tex const& tex, vector<2, FloatT> coord)
    -> decltype( tex2D_impl_expand_types(
            tex.data(),
            coord,
            vector<2, FloatT>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 2, "Incompatible texture type");

    vector<2, FloatT> texsize( tex.width(), tex.height() );

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
