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
    typename FloatT
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<1, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<1, F> texsizef(F((float)texsize[0]));

    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord[0] * texsizef[0] - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto tmp0 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h0   = ( floorx - FloatT(0.5) + tmp0 ) / texsizef[0];

    auto tmp1 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h1   = ( floorx + FloatT(1.5) + tmp1 ) / texsizef[0];

    auto f_0  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<1, FloatT>(h0)) );
    auto f_1  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<1, FloatT>(h1)) );

    return ReturnT(g0(fracx) * f_0 + g1(fracx) * f_1);
}


//-------------------------------------------------------------------------------------------------
// 2D
//

template <
    typename ReturnT,
    typename InternalT,
    typename Tex,
    typename FloatT
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<2, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<2, F> texsizef(F((float)texsize[0]), F((float)texsize[1]));

    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord.x * texsizef[0] - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * texsizef[1] - FloatT(0.5);
    auto floory = floor(y);
    auto fracy  = y - floor(y);


    auto tmp00 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_00  = ( floorx - FloatT(0.5) + tmp00 ) / texsizef[0];

    auto tmp10 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_10  = ( floorx + FloatT(1.5) + tmp10 ) / texsizef[0];

    auto tmp01 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_01  = ( floory - FloatT(0.5) + tmp01 ) / texsizef[1];

    auto tmp11 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_11  = ( floory + FloatT(1.5) + tmp11 ) / texsizef[1];


    auto f_00  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<2, FloatT>(h_00, h_01)) );
    auto f_10  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<2, FloatT>(h_10, h_01)) );
    auto f_01  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<2, FloatT>(h_00, h_11)) );
    auto f_11  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<2, FloatT>(h_10, h_11)) );

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
    typename FloatT
    >
inline ReturnT cubic_opt(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<3, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<3, F> texsizef(F((float)texsize[0]), F((float)texsize[1]), F((float)texsize[2]));

    bspline::w0_func w0;
    bspline::w1_func w1;
    bspline::w2_func w2;
    bspline::w3_func w3;

    auto x = coord.x * texsizef[0] - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto y = coord.y * texsizef[1] - FloatT(0.5);
    auto floory = floor(y);
    auto fracy  = y - floor(y);

    auto z = coord.z * texsizef[2] - FloatT(0.5);
    auto floorz = floor(z);
    auto fracz  = z - floor(z);


    auto tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h_000  = ( floorx - FloatT(0.5) + tmp000 ) / texsizef[0];

    auto tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h_100  = ( floorx + FloatT(1.5) + tmp100 ) / texsizef[0];

    auto tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    auto h_010  = ( floory - FloatT(0.5) + tmp010 ) / texsizef[1];

    auto tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    auto h_110  = ( floory + FloatT(1.5) + tmp110 ) / texsizef[1];

    auto tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    auto h_001  = ( floorz - FloatT(0.5) + tmp001 ) / texsizef[2];

    auto tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    auto h_101  = ( floorz + FloatT(1.5) + tmp101 ) / texsizef[2];


    auto f_000  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_000, h_010, h_001)) );
    auto f_100  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_100, h_010, h_001)) );
    auto f_010  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_000, h_110, h_001)) );
    auto f_110  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_100, h_110, h_001)) );

    auto f_001  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_000, h_010, h_101)) );
    auto f_101  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_100, h_010, h_101)) );
    auto f_011  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_000, h_110 ,h_101)) );
    auto f_111  = InternalT( linear(ReturnT{}, InternalT{}, tex, vector<3, FloatT>(h_100, h_110, h_101)) );

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
