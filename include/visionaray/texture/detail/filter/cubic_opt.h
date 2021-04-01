// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_CUBIC_OPT_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_CUBIC_OPT_H 1

#include <visionaray/math/detail/math.h>
#include <visionaray/math/vector.h>

#include "common.h"
#include "linear.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// 1D
//

template <
    typename ReturnT,
    typename InternalT,
    typename Tex,
    typename TexelT,
    typename FloatT,
    typename TexSize
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<1, FloatT> coord,
        TexSize           texsize
        )
{
    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord[0] * FloatT(texsize[0]) - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto tmp0 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h0   = ( floorx - FloatT(0.5) + tmp0 ) / FloatT(texsize[0]);

    auto tmp1 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h1   = ( floorx + FloatT(1.5) + tmp1 ) / FloatT(texsize[0]);

    auto f_0  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<1, FloatT>(h0), texsize) );
    auto f_1  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<1, FloatT>(h1), texsize) );

    return ReturnT(g0(fracx) * f_0 + g1(fracx) * f_1);
}


//-------------------------------------------------------------------------------------------------
// 2D
//

template <
    typename ReturnT,
    typename InternalT,
    typename Tex,
    typename TexelT,
    typename FloatT,
    typename TexSize
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<2, FloatT> coord,
        TexSize           texsize
        )
{
    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord.x * FloatT(texsize.x) - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * FloatT(texsize.y) - FloatT(0.5);
    auto floory = floor(y);
    auto fracy  = y - floor(y);


    auto tmp00 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_00  = ( floorx - FloatT(0.5) + tmp00 ) / FloatT(texsize.x);

    auto tmp10 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_10  = ( floorx + FloatT(1.5) + tmp10 ) / FloatT(texsize.x);

    auto tmp01 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_01  = ( floory - FloatT(0.5) + tmp01 ) / FloatT(texsize.y);

    auto tmp11 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_11  = ( floory + FloatT(1.5) + tmp11 ) / FloatT(texsize.y);


    auto f_00  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<2, FloatT>(h_00, h_01), texsize) );
    auto f_10  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<2, FloatT>(h_10, h_01), texsize) );
    auto f_01  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<2, FloatT>(h_00, h_11), texsize) );
    auto f_11  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<2, FloatT>(h_10, h_11), texsize) );

    auto f_0   = g0(fracx) * f_00 + g1(fracx) * f_10;
    auto f_1   = g0(fracx) * f_01 + g1(fracx) * f_11;

    return ReturnT(g0(fracy) * f_0 + g1(fracy) * f_1);
}


//-------------------------------------------------------------------------------------------------
// 3D
//

template <
    typename ReturnT,
    typename InternalT,
    typename Tex,
    typename TexelT,
    typename FloatT,
    typename TexSize
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<3, FloatT> coord,
        TexSize           texsize
        )
{
    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord.x * FloatT(texsize.x) - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * FloatT(texsize.y) - FloatT(0.5);
    auto floory = floor(y);
    auto fracy  = y - floor(y);

    auto z = coord.z * FloatT(texsize.z) - FloatT(0.5);
    auto floorz = floor(z);
    auto fracz  = z - floor(z);


    auto tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_000  = ( floorx - FloatT(0.5) + tmp000 ) / FloatT(texsize.x);

    auto tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_100  = ( floorx + FloatT(1.5) + tmp100 ) / FloatT(texsize.x);

    auto tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_010  = ( floory - FloatT(0.5) + tmp010 ) / FloatT(texsize.y);

    auto tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_110  = ( floory + FloatT(1.5) + tmp110 ) / FloatT(texsize.y);

    auto tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    auto h_001  = ( floorz - FloatT(0.5) + tmp001 ) / FloatT(texsize.z);

    auto tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    auto h_101  = ( floorz + FloatT(1.5) + tmp101 ) / FloatT(texsize.z);


    auto f_000  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_000, h_010, h_001), texsize) );
    auto f_100  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_100, h_010, h_001), texsize) );
    auto f_010  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_000, h_110, h_001), texsize) );
    auto f_110  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_100, h_110, h_001), texsize) );

    auto f_001  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_000, h_010, h_101), texsize) );
    auto f_101  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_100, h_010, h_101), texsize) );
    auto f_011  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_000, h_110 ,h_101), texsize) );
    auto f_111  = InternalT( linear(ReturnT{}, InternalT{}, tex, ptr, vector<3, FloatT>(h_100, h_110, h_101), texsize) );

    auto f_00   = g0(fracx) * f_000 + g1(fracx) * f_100;
    auto f_10   = g0(fracx) * f_010 + g1(fracx) * f_110;
    auto f_01   = g0(fracx) * f_001 + g1(fracx) * f_101;
    auto f_11   = g0(fracx) * f_011 + g1(fracx) * f_111;

    auto f_0    = g0(fracy) * f_00 + g1(fracy) * f_10;
    auto f_1    = g0(fracy) * f_01 + g1(fracy) * f_11;

    return ReturnT(g0(fracz) * f_0 + g1(fracz) * f_1);
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_CUBIC_OPT_H
