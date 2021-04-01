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
    typename FloatT
    >
inline ReturnT choose_filter(
        ReturnT       /* */,
        InternalT     /* */,
        Tex const&    tex,
        FloatT        coord
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
                coord
                );

    case visionaray::Linear:
        return linear(
                ReturnT{},
                InternalT{},
                tex,
                coord
                );

    case visionaray::BSpline:
        return cubic_opt(
                ReturnT{},
                InternalT{},
                tex,
                coord
                );

    case visionaray::CardinalSpline:
        return cubic(
                ReturnT{},
                InternalT{},
                tex,
                coord,
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
