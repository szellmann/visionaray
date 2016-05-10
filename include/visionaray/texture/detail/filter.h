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
    typename TexelT,
    typename FloatT,
    typename SizeT,
    typename AddressMode
    >
inline ReturnT choose_filter(
        ReturnT             /* */,
        InternalT           /* */,
        TexelT const&       tex,
        FloatT              coord,
        SizeT               texsize,
        tex_filter_mode     filter_mode,
        AddressMode const&  address_mode
        )
{
    switch (filter_mode)
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest(
                ReturnT{},
                InternalT{},
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::Linear:
        return linear(
                ReturnT{},
                InternalT{},
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::BSpline:
        return cubic_opt(
                ReturnT{},
                InternalT{},
                tex,
                coord,
                texsize,
                address_mode
                );

/*    case visionaray::CardinalSpline:
        return cubic(
                ReturnT{},
                InternalT{},
                tex,
                coord,
                texsize,
                address_mode,
                cspline::w0_func<FloatT>(),
                cspline::w1_func<FloatT>(),
                cspline::w2_func<FloatT>(),
                cspline::w3_func<FloatT>()
                );*/

    }
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_FILTER_H
