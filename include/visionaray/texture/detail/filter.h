// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_FILTER_H
#define VSNRAY_TEXTURE_DETAIL_FILTER_H 1

#include "filter/cubic.h"
#include "filter/cubic_opt.h"
#include "filter/linear.h"
#include "filter/nearest.h"

#include "texture_common.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <
    typename ReturnT,
    typename InternalT,
    typename Tex,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT choose_filter(
        ReturnT       /* */,
        InternalT     /* */,
        Tex const&    tex,
        TexelT const& ptr,
        FloatT        coord,
        SizeT         texsize
        )
{
    switch (tex.get_filter_mode())
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest(
                ReturnT{},
                InternalT{},
                tex,
                ptr,
                coord,
                texsize
                );

    case visionaray::Linear:
        return linear(
                ReturnT{},
                InternalT{},
                tex,
                ptr,
                coord,
                texsize
                );

    case visionaray::BSpline:
        return cubic_opt(
                ReturnT{},
                InternalT{},
                tex,
                ptr,
                coord,
                texsize
                );

    case visionaray::CardinalSpline:
        return cubic(
                ReturnT{},
                InternalT{},
                tex,
                ptr,
                coord,
                texsize,
                cspline::w0_func(),
                cspline::w1_func(),
                cspline::w2_func(),
                cspline::w3_func()
                );

    }
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_H
