// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER1D_H
#define VSNRAY_TEXTURE_SAMPLER1D_H

#include <visionaray/math/math.h>

#include "sampler_common.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT nearest(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        FloatT                                  texsize,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    FloatT lo = floor(coord * texsize);
    lo = clamp(lo, FloatT(0.0f), texsize - 1);
    return point(tex, lo, ReturnT());
}


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT linear(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        FloatT                                  texsize,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    FloatT texcoordf( coord * texsize - FloatT(0.5) );
    texcoordf = clamp( texcoordf, FloatT(0.0), texsize - 1 );

    FloatT lo = floor(texcoordf);
    FloatT hi = ceil(texcoordf);


    InternalT samples[2] =
    {
        InternalT( point(tex, lo, ReturnT()) ),
        InternalT( point(tex, hi, ReturnT()) )
    };

    auto u = texcoordf - lo;

    return lerp( samples[0], samples[1], u );
}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT cubic2(TexelT const* tex, FloatT coord, FloatT texsize)
{

    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;

    FloatT x = coord * texsize - FloatT(0.5);
    FloatT floorx = floor( x );
    FloatT fracx  = x - floor( x );

    FloatT tmp0 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    FloatT h0   = ( floorx - FloatT(0.5) + tmp0 ) / texsize;

    FloatT tmp1 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    FloatT h1   = ( floorx + FloatT(1.5) + tmp1 ) / texsize;


    // In visionaray, the return type is a float4.
    // TODO: what if precision(ReturnT) < precision(FloatT)?
    ReturnT f_0 = linear<ReturnT>( tex, h0, texsize, Clamp /* TODO */ );
    ReturnT f_1 = linear<ReturnT>( tex, h1, texsize, Clamp /* TODO */ );

    return g0(fracx) * f_0 + g1(fracx) * f_1;

}


template <typename ReturnT, typename W0, typename W1, typename W2, typename W3, typename FloatT, typename TexelT>
inline ReturnT cubic(TexelT const* tex, FloatT coord, FloatT texsize, W0 w0, W1 w1, W2 w2, W3 w3)
{

    FloatT x = coord * texsize - FloatT(0.5);
    FloatT floorx = floor( x );
    FloatT fracx = x - floor( x );

    FloatT pos[4] =
    {
        FloatT( floorx - 1 ),
        FloatT( floorx ),
        FloatT( floorx + 1 ),
        FloatT( floorx + 2 )
    };

    for (size_t i = 0; i < 4; ++i)
    {
        pos[i] = clamp(pos[i], FloatT(0.0), texsize - FloatT(1.0));
    }

#define TEX(x) (point(tex,x, ReturnT()))
    return w0(fracx) * TEX(pos[0]) + w1(fracx) * TEX(pos[1]) + w2(fracx) * TEX(pos[2]) + w3(fracx) * TEX(pos[3]);
#undef TEX
}


//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <typename ReturnT, typename FloatT, typename TexelT, typename InternalT>
inline ReturnT tex1D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const&                           tex,
        FloatT                                  coord,
        FloatT                                  texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
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
inline T tex1D_impl_expand_types(
        T const*                                tex,
        float                                   coord,
        float                                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = float;

    return tex1D_impl_choose_filter(
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
inline vector<4, simd::float4> tex1D_impl_expand_types(
        vector<4, T> const*                     tex,
        simd::float4 const&                     coord,
        simd::float4 const&                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = vector<4, simd::float4>;
    using internal_type = vector<4, simd::float4>;

    return tex1D_impl_choose_filter(
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
// tex1D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex1D(Tex const& tex, FloatT coord)
    -> decltype( tex1D_impl_expand_types(
            tex.data(),
            coord,
            FloatT(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    auto texsize = FloatT( tex.width() );

    return tex1D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}


} // detail
} // visionaray


#endif // VSNRAY_TEXTURE_SAMPLER1D_H
