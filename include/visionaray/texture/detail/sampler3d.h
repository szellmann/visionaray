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


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT nearest(TexelT const* tex, vector<3, FloatT> coord, vector<3, FloatT> texsize)
{

#if 1

    using visionaray::clamp;

    typedef vector<3, FloatT> float3;

    float3 lo
    (
        floor(coord.x * texsize.x),
        floor(coord.y * texsize.y),
        floor(coord.z * texsize.z)
    );

    lo[0] = clamp(lo[0], FloatT(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], FloatT(0.0f), texsize[1] - 1);
    lo[2] = clamp(lo[2], FloatT(0.0f), texsize[2] - 1);

    FloatT idx = index(lo[0], lo[1], lo[2], texsize);
    return point(tex, idx);

#else

    // TODO: should be done similar to the code below..

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - FloatT(0.5),
                      coorm[1] * texsizef[1] - FloatT(0.5),
                      coord[2] * texsizef[2] - FloatT(0.5));

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsizef[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsizef[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsizef[2] - 1 );

    Float3T lof( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    Float3T hif( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );
    Int3T   lo( ftoi(lof[0]), ftoi(lof[1]), ftoi(lof[2]) );
    Int3T   hi( ftoi(hif[0]), ftoi(hif[1]), ftoi(hif[2]) );

    Float3T uvw = texcoordf - uvw;

    IntT idx = index( uvw[0] < FloatT(0.5) ? lo[0] : hi[0],
                      uvw[1] < FloatT(0.5) ? lo[1] : hi[1],
                      uvw[2] < FloatT(0.5) ? lo[2] : hi[2],
                      texsize);

    return point(tex, idx);

#endif

}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT linear(TexelT const* tex, vector<3, FloatT> coord, vector<3, FloatT> texsize)
{

    using visionaray::clamp;

    typedef vector<3, FloatT> float3;

    float3 texcoordf( coord * texsize - FloatT(0.5) );

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsize[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsize[2] - 1 );

    float3 lo( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    float3 hi( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    FloatT samples[8] =
    {
        FloatT( point(tex, index( lo.x, lo.y, lo.z, texsize )) ),
        FloatT( point(tex, index( hi.x, lo.y, lo.z, texsize )) ),
        FloatT( point(tex, index( lo.x, hi.y, lo.z, texsize )) ),
        FloatT( point(tex, index( hi.x, hi.y, lo.z, texsize )) ),
        FloatT( point(tex, index( lo.x, lo.y, hi.z, texsize )) ),
        FloatT( point(tex, index( hi.x, lo.y, hi.z, texsize )) ),
        FloatT( point(tex, index( lo.x, hi.y, hi.z, texsize )) ),
        FloatT( point(tex, index( hi.x, hi.y, hi.z, texsize )) )
    };


    float3 uvw = texcoordf - lo;

    FloatT p1  = lerp(samples[0], samples[1], uvw[0]);
    FloatT p2  = lerp(samples[2], samples[3], uvw[0]);
    FloatT p3  = lerp(samples[4], samples[5], uvw[0]);
    FloatT p4  = lerp(samples[6], samples[7], uvw[0]);

    FloatT p12 = lerp(p1, p2, uvw[1]);
    FloatT p34 = lerp(p3, p4, uvw[1]);

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
    auto f_000 = linear<FloatT>( tex, float3(h_000, h_010, h_001), texsize );
    auto f_100 = linear<FloatT>( tex, float3(h_100, h_010, h_001), texsize );
    auto f_010 = linear<FloatT>( tex, float3(h_000, h_110, h_001), texsize );
    auto f_110 = linear<FloatT>( tex, float3(h_100, h_110, h_001), texsize );

    auto f_001 = linear<FloatT>( tex, float3(h_000, h_010, h_101), texsize );
    auto f_101 = linear<FloatT>( tex, float3(h_100, h_010, h_101), texsize );
    auto f_011 = linear<FloatT>( tex, float3(h_000, h_110 ,h_101), texsize );
    auto f_111 = linear<FloatT>( tex, float3(h_100, h_110, h_101), texsize );

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

#define TEX(x,y,z) (point(tex, index(x,y,z,texsize)))
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


template<typename ReturnT, typename Tex, typename FloatT>
inline ReturnT tex3D(Tex const& tex, vector<3, FloatT> coord)
{

    static_assert(Tex::dimensions == 3, "Incompatible texture type");

    vector<3, FloatT> texsize( tex.width(), tex.height(), tex.depth() );

    switch (tex.get_filter_mode())
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest<ReturnT>( tex.data(), coord, texsize );

    case visionaray::Linear:
        return linear<ReturnT>( tex.data(), coord, texsize );

    case visionaray::BSpline:
        return cubic8<ReturnT>( tex.data(), coord, texsize );

    case visionaray::BSplineInterpol:
        return cubic<ReturnT>( tex.prefiltered_data, coord, texsize,
            bspline::w0_func<FloatT>(), bspline::w1_func<FloatT>(),
            bspline::w2_func<FloatT>(), bspline::w3_func<FloatT>() );


    case visionaray::CardinalSpline:
        return cubic<ReturnT>( tex.data(), coord, texsize,
            cspline::w0_func<FloatT>(), cspline::w1_func<FloatT>(),
            cspline::w2_func<FloatT>(), cspline::w3_func<FloatT>() );

    }

}


} // detail
} // visionaray


#endif // VSNRAY_TEXTURE_SAMPLER3D_H
