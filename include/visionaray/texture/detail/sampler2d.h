// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER2D_H
#define VSNRAY_TEXTURE_SAMPLER2D_H

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

    using visionaray::clamp;

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

    }
}


//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

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


//-------------------------------------------------------------------------------------------------
// tex2D() dispatch function
//

template <typename ReturnT, typename Tex, typename FloatT>
inline ReturnT tex2D(Tex const& tex, vector<2, FloatT> coord)
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
