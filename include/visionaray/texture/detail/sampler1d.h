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


template <typename ReturnT, typename FloatT, typename VoxelT>
inline ReturnT nearest(VoxelT const* tex, FloatT coord, FloatT texsize)
{

    FloatT lo = floor(coord * texsize);
    lo = clamp(lo, FloatT(0.0f), texsize - 1);
    return point(tex, lo);

}


template <typename ReturnT, typename FloatT, typename VoxelT>
inline ReturnT linear(VoxelT const* tex, FloatT coord, FloatT texsize)
{

    FloatT texcoordf( coord * texsize - FloatT(0.5) );
    texcoordf = clamp( texcoordf, FloatT(0.0), texsize - 1 );

    FloatT lo = floor(texcoordf);
    FloatT hi = ceil(texcoordf);


    // In visionaray, the return type is a float4.
    // TODO: what if precision(ReturnT) < precision(FloatT)?
    ReturnT samples[2] =
    {
        point(tex, lo),
        point(tex, hi)
    };

    FloatT u = texcoordf - lo;

    return lerp( samples[0], samples[1], u );

}


template <typename ReturnT, typename FloatT, typename VoxelT>
inline ReturnT cubic2(VoxelT const* tex, FloatT coord, FloatT texsize)
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
    ReturnT f_0 = linear<ReturnT>( tex, h0, texsize );
    ReturnT f_1 = linear<ReturnT>( tex, h1, texsize );

    return g0(fracx) * f_0 + g1(fracx) * f_1;

}


template <typename ReturnT, typename W0, typename W1, typename W2, typename W3, typename FloatT, typename VoxelT>
inline ReturnT cubic(VoxelT const* tex, FloatT coord, FloatT texsize, W0 w0, W1 w1, W2 w2, W3 w3)
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

#define TEX(x) (point(tex,x))
    return w0(fracx) * TEX(pos[0]) + w1(fracx) * TEX(pos[1]) + w2(fracx) * TEX(pos[2]) + w3(fracx) * TEX(pos[3]);
#undef TEX
}


template <typename ReturnT, typename FloatT, typename VoxelT>
inline ReturnT tex1D(texture<VoxelT, visionaray::ElementType, 1> const& tex, FloatT coord)
{

    FloatT texsize = tex.width();

    switch (tex.get_filter_mode())
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest<ReturnT>( tex.data, coord, texsize );

    case visionaray::Linear:
        return linear<ReturnT>( tex.data, coord, texsize );

    case visionaray::BSpline:
        return cubic2<ReturnT>( tex.data, coord, texsize );

/*    case visionaray::BSplineInterpol:
        return cubic<ReturnT>( tex.prefiltered_data, coord, texsize,
            bspline::w0_func<FloatT>(), bspline::w1_func<FloatT>(),
            bspline::w2_func<FloatT>(), bspline::w3_func<FloatT>() );*/


    case visionaray::CardinalSpline:
        return cubic<ReturnT>( tex.data, coord, texsize,
            cspline::w0_func<FloatT>(), cspline::w1_func<FloatT>(),
            cspline::w2_func<FloatT>(), cspline::w3_func<FloatT>() );

    }

}


} // detail
} // visionaray


#endif // VSNRAY_TEXTURE_SAMPLER1D_H


