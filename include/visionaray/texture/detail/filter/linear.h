// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_LINEAR_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_LINEAR_H 1

#include <visionaray/math/detail/math.h>
#include <visionaray/math/vector.h>

#include "common.h"

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
VSNRAY_FUNC
inline ReturnT linear(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<1, FloatT> coord
        )
{
    using F = FloatT;
    using I = decltype(convert_to_int(FloatT{}));
    auto texsize = tex.size();
    vector<1, F> texsizef(F((float)texsize[0]));
    vector<1, I> texsize_minus_one(texsize[0] - 1);

    auto coord1 = tex.remap_texture_coordinate(coord - FloatT(0.5) / texsizef);
    auto coord2 = tex.remap_texture_coordinate(coord + FloatT(0.5) / texsizef);

    auto lo = min(convert_to_int(coord1 * texsizef), texsize_minus_one);
    auto hi = min(convert_to_int(coord2 * texsizef), texsize_minus_one);

    InternalT samples[2] = {
        InternalT(tex.value(ReturnT{}, lo[0])),
        InternalT(tex.value(ReturnT{}, hi[0]))
        };

    auto u = coord1[0] * FloatT(texsize[0]) - FloatT(lo[0]);

    return ReturnT(lerp_r(samples[0], samples[1], u));
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
VSNRAY_FUNC
inline ReturnT linear(
        ReturnT                  /* */,
        InternalT                /* */,
        Tex const&               tex,
        vector<2, FloatT> const& coord
        )
{
    using F = FloatT;
    using I = decltype(convert_to_int(FloatT{}));
    auto texsize = tex.size();
    vector<2, F> texsizef(F((float)texsize[0]), F((float)texsize[1]));
    vector<2, I> texsize_minus_one(texsize[0] - 1, texsize[1] - 1);

    auto coord1 = tex.remap_texture_coordinate(coord - FloatT(0.5) / texsizef);
    auto coord2 = tex.remap_texture_coordinate(coord + FloatT(0.5) / texsizef);

    auto lo = min(convert_to_int(coord1 * texsizef), texsize_minus_one);
    auto hi = min(convert_to_int(coord2 * texsizef), texsize_minus_one);

    InternalT samples[4] = {
        InternalT(tex.value(ReturnT{}, lo.x, lo.y)),
        InternalT(tex.value(ReturnT{}, hi.x, lo.y)),
        InternalT(tex.value(ReturnT{}, lo.x, hi.y)),
        InternalT(tex.value(ReturnT{}, hi.x, hi.y))
        };


    auto uv = coord1 * texsizef - vector<2, FloatT>(lo);

    auto p1 = lerp_r(samples[0], samples[1], uv[0]);
    auto p2 = lerp_r(samples[2], samples[3], uv[0]);

    return ReturnT(lerp_r(p1, p2, uv[1]));
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
VSNRAY_FUNC
inline ReturnT linear(
        ReturnT                  /* */,
        InternalT                /* */,
        Tex const&               tex,
        vector<3, FloatT> const& coord
        )
{
    using F = FloatT;
    using I = decltype(convert_to_int(FloatT{}));
    auto texsize = tex.size();
    vector<3, F> texsizef(F((float)texsize[0]), F((float)texsize[1]), F((float)texsize[2]));
    vector<3, I> texsize_minus_one(texsize[0] - 1, texsize[1] - 1, texsize[2] - 1);

    auto coord1 = tex.remap_texture_coordinate(coord - FloatT(0.5) / texsizef);
    auto coord2 = tex.remap_texture_coordinate(coord + FloatT(0.5) / texsizef);

    auto lo = min(convert_to_int(coord1 * texsizef), texsize_minus_one);
    auto hi = min(convert_to_int(coord2 * texsizef), texsize_minus_one);

    InternalT samples[8] = {
        InternalT(tex.value(ReturnT{}, lo.x, lo.y, lo.z)),
        InternalT(tex.value(ReturnT{}, hi.x, lo.y, lo.z)),
        InternalT(tex.value(ReturnT{}, lo.x, hi.y, lo.z)),
        InternalT(tex.value(ReturnT{}, hi.x, hi.y, lo.z)),
        InternalT(tex.value(ReturnT{}, lo.x, lo.y, hi.z)),
        InternalT(tex.value(ReturnT{}, hi.x, lo.y, hi.z)),
        InternalT(tex.value(ReturnT{}, lo.x, hi.y, hi.z)),
        InternalT(tex.value(ReturnT{}, hi.x, hi.y, hi.z))
        };


    auto uvw = coord1 * texsizef - vector<3, FloatT>(lo);

    auto p1  = lerp_r(samples[0], samples[1], uvw[0]);
    auto p2  = lerp_r(samples[2], samples[3], uvw[0]);
    auto p3  = lerp_r(samples[4], samples[5], uvw[0]);
    auto p4  = lerp_r(samples[6], samples[7], uvw[0]);

    auto p12 = lerp_r(p1, p2, uvw[1]);
    auto p34 = lerp_r(p3, p4, uvw[1]);

    return ReturnT(lerp_r(p12, p34, uvw[2]));
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_LINEAR_H
