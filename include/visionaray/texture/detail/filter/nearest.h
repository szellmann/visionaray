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
    typename FloatT
    >
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<1, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<1, F> texsizef(F((float)texsize[0]));

    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord * texsizef);
    return point(tex.data(), lo[0], ReturnT{});
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
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<2, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<2, F> texsizef(F((float)texsize[0]), F((float)texsize[1]));

    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord * texsizef);

    auto idx = linear_index(lo[0], lo[1], texsize);
    return point(tex.data(), idx, ReturnT{});
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
inline ReturnT nearest(
        ReturnT           /* */,
        InternalT         /* */,
        Tex const&        tex,
        vector<3, FloatT> coord
        )
{
    using F = FloatT;
    auto texsize = tex.size();
    vector<3, F> texsizef(F((float)texsize[0]), F((float)texsize[1]), F((float)texsize[2]));

    coord = tex.remap_texture_coordinate(coord);

    auto lo = convert_to_int(coord * texsizef);

    auto idx = linear_index(lo[0], lo[1], lo[2], texsize);
    return point(tex.data(), idx, ReturnT{});
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_NEAREST_H
