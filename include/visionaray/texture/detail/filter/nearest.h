// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_NEAREST_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_NEAREST_H 1

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
    typename TexelT,
    typename FloatT,
    typename TexSize
    >
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<1, FloatT> coord,
        TexSize           texsize
        )
{
    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord[0] * FloatT(texsize[0]));
    return point(ptr, lo, ReturnT{});
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
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<2, FloatT> coord,
        TexSize           texsize
        )
{
    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord * vector<2, FloatT>(texsize));

    auto idx = linear_index(lo[0], lo[1], texsize);
    return point(ptr, idx, ReturnT{});
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
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        TexelT const*     ptr,
        vector<3, FloatT> coord,
        TexSize           texsize
        )
{
    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord * vector<3, FloatT>(texsize));

    auto idx = linear_index(lo[0], lo[1], lo[2], texsize);
    return point(ptr, idx, ReturnT{});
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_NEAREST_H
