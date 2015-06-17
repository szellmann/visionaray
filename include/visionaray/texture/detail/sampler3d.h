// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER3D_H
#define VSNRAY_TEXTURE_SAMPLER3D_H

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


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT nearest(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<3, FloatT>                       coord,
        vector<3, FloatT>                       texsize,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    vector<3, FloatT> lo(
            floor(coord.x * texsize.x),
            floor(coord.y * texsize.y),
            floor(coord.z * texsize.z)
            );

    lo[0] = clamp(lo[0], FloatT(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], FloatT(0.0f), texsize[1] - 1);
    lo[2] = clamp(lo[2], FloatT(0.0f), texsize[2] - 1);

    FloatT idx = index(lo[0], lo[1], lo[2], texsize);
    return point(tex, idx, ReturnT());
}


template <typename ReturnT, typename InternalT, typename FloatT, typename TexelT>
inline ReturnT linear(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<3, FloatT>                       coord,
        vector<3, FloatT>                       texsize,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    coord = map_tex_coord(coord, address_mode);

    vector<3, FloatT> texcoordf( coord * texsize - FloatT(0.5) );

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsize[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsize[2] - 1 );

    vector<3, FloatT> lo( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    vector<3, FloatT> hi( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );


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


    auto uvw = texcoordf - lo;

    auto p1  = lerp(samples[0], samples[1], uvw[0]);
    auto p2  = lerp(samples[2], samples[3], uvw[0]);
    auto p3  = lerp(samples[4], samples[5], uvw[0]);
    auto p4  = lerp(samples[6], samples[7], uvw[0]);

    auto p12 = lerp(p1, p2, uvw[1]);
    auto p34 = lerp(p3, p4, uvw[1]);

    return lerp(p12, p34, uvw[2]);
}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT cubic8(TexelT const* tex, vector<3, FloatT> coord, vector<3, FloatT> texsize)
{

    typedef vector<3, FloatT> float3;

    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;

    auto x = coord.x * texsize.x - FloatT(0.5);
    auto floorx = floor( x );
    auto fracx  = x - floor( x );

    auto y = coord.y * texsize.y - FloatT(0.5);
    auto floory = floor( y );
    auto fracy  = y - floor( y );

    auto z = coord.z * texsize.z - FloatT(0.5);
    auto floorz = floor( z );
    auto fracz  = z - floor( z );


    auto tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_000  = ( floorx - FloatT(0.5) + tmp000 ) / texsize.x;

    auto tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_100  = ( floorx + FloatT(1.5) + tmp100 ) / texsize.x;

    auto tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_010  = ( floory - FloatT(0.5) + tmp010 ) / texsize.y;

    auto tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_110  = ( floory + FloatT(1.5) + tmp110 ) / texsize.y;

    auto tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    auto h_001  = ( floorz - FloatT(0.5) + tmp001 ) / texsize.z;

    auto tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    auto h_101  = ( floorz + FloatT(1.5) + tmp101 ) / texsize.z;


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    auto f_000 = linear<FloatT>( tex, float3(h_000, h_010, h_001), texsize, Clamp /* TODO */ );
    auto f_100 = linear<FloatT>( tex, float3(h_100, h_010, h_001), texsize, Clamp /* TODO */ );
    auto f_010 = linear<FloatT>( tex, float3(h_000, h_110, h_001), texsize, Clamp /* TODO */ );
    auto f_110 = linear<FloatT>( tex, float3(h_100, h_110, h_001), texsize, Clamp /* TODO */ );

    auto f_001 = linear<FloatT>( tex, float3(h_000, h_010, h_101), texsize, Clamp /* TODO */ );
    auto f_101 = linear<FloatT>( tex, float3(h_100, h_010, h_101), texsize, Clamp /* TODO */ );
    auto f_011 = linear<FloatT>( tex, float3(h_000, h_110 ,h_101), texsize, Clamp /* TODO */ );
    auto f_111 = linear<FloatT>( tex, float3(h_100, h_110, h_101), texsize, Clamp /* TODO */ );

    auto f_00  = g0(fracx) * f_000 + g1(fracx) * f_100;
    auto f_10  = g0(fracx) * f_010 + g1(fracx) * f_110;
    auto f_01  = g0(fracx) * f_001 + g1(fracx) * f_101;
    auto f_11  = g0(fracx) * f_011 + g1(fracx) * f_111;

    auto f_0   = g0(fracy) * f_00 + g1(fracy) * f_10;
    auto f_1   = g0(fracy) * f_01 + g1(fracy) * f_11;

    return g0(fracz) * f_0 + g1(fracz) * f_1;

}


template <typename ReturnT, typename W0, typename W1, typename W2, typename W3, typename FloatT, typename TexelT>
inline ReturnT cubic(TexelT const* tex, vector<3, FloatT> coord, vector<3, FloatT> texsize, W0 w0, W1 w1, W2 w2, W3 w3)
{

    typedef vector<3, FloatT> float3;

    auto x = coord.x * texsize.x - FloatT(0.5);
    auto floorx = floor( x );
    auto fracx  = x - floor( x );

    auto y = coord.y * texsize.y - FloatT(0.5);
    auto floory = floor( y );
    auto fracy  = y - floor( y );

    auto z = coord.z * texsize.z - FloatT(0.5);
    auto floorz = floor( z );
    auto fracz  = z - floor( z );

    float3 pos[4] =
    {
        float3( floorx - 1, floory - 1, floorz - 1 ),
        float3( floorx,     floory,     floorz ),
        float3( floorx + 1, floory + 1, floorz + 1 ),
        float3( floorx + 2, floory + 2, floorz + 2 )
    };

    using visionaray::clamp;

    for (size_t i = 0; i < 4; ++i)
    {
        pos[i].x = clamp(pos[i].x, FloatT(0.0), texsize.x - 1);
        pos[i].y = clamp(pos[i].y, FloatT(0.0), texsize.y - 1);
        pos[i].z = clamp(pos[i].z, FloatT(0.0), texsize.z - 1);
    }

#define TEX(x,y,z) (point(tex, index(x,y,z,texsize), ReturnT()))
    auto f00 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[0].z);
    auto f01 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[0].z);
    auto f02 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[0].z);
    auto f03 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[0].z);

    auto f04 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[1].z);
    auto f05 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[1].z);
    auto f06 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[1].z);
    auto f07 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[1].z);

    auto f08 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[2].z);
    auto f09 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[2].z);
    auto f10 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[2].z);
    auto f11 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[2].z);

    auto f12 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[3].z);
    auto f13 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[3].z);
    auto f14 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[3].z);
    auto f15 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[3].z);
#undef TEX

    auto f0 = w0(fracy) * f00 + w1(fracy) * f01 + w2(fracy) * f02 + w3(fracy) * f03;
    auto f1 = w0(fracy) * f04 + w1(fracy) * f05 + w2(fracy) * f06 + w3(fracy) * f07;
    auto f2 = w0(fracy) * f08 + w1(fracy) * f09 + w2(fracy) * f10 + w3(fracy) * f11;
    auto f3 = w0(fracy) * f12 + w1(fracy) * f13 + w2(fracy) * f14 + w3(fracy) * f15;

    return w0(fracz) * f0 + w1(fracz) * f1 + w2(fracz) * f2 + w3(fracz) * f3;

}


//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <typename ReturnT, typename FloatT, typename TexelT, typename InternalT>
inline ReturnT tex3D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        vector<3, FloatT> const&                coord,
        vector<3, FloatT> const&                texsize,
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
    }
}


//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

template <typename T>
inline T tex3D_impl_expand_types(
        T const*                                tex,
        vector<3, float> const&                 coord,
        vector<3, float> const&                 texsize,
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

template <typename T>
inline vector<3, T> tex3D_impl_expand_types(
        vector<3, T> const*                     tex,
        vector<3, float> const&                 coord,
        vector<3, float> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = vector<3, T>;
    using internal_type = vector<3, float>;

    tex3D_impl_choose_filter(
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
inline vector<4, T> tex3D_impl_expand_types(
        vector<4, T> const*                     tex,
        vector<3, float> const&                 coord,
        vector<3, float> const&                 texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = vector<4, T>;
    using internal_type = vector<4, float>;

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


template <typename T>
inline simd::float4 tex3D_impl_expand_types(
        T const*                                tex,
        vector<3, simd::float4> const&          coord,
        vector<3, simd::float4> const&          texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 3> const&  address_mode
        )
{
    using return_type   = simd::float4;
    using internal_type = simd::float4;

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
            vector<3, FloatT>(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    vector<3, FloatT> texsize( tex.width(), tex.height(), tex.depth() );

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
