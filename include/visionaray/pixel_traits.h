// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PIXEL_TRAITS_H
#define VSNRAY_PIXEL_TRAITS_H

#include <visionaray/math/math.h>
#include <visionaray/norm.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{

template <pixel_format PF>
struct pixel_traits
{
    /* No pixel_traits::{format()|type} causes a compiler error */
};

template <>
struct pixel_traits<PF_UNSPECIFIED>
{
    VSNRAY_FUNC constexpr static pixel_format format() { return PF_UNSPECIFIED; }
    /* Note: no type! */
};

template <>
struct pixel_traits<PF_RGB8>
{
    VSNRAY_FUNC constexpr static pixel_format format() { return PF_RGB8; }
    typedef vector<3, unorm< 8>> type;
};

template <>
struct pixel_traits<PF_RGBA8>
{
    VSNRAY_FUNC constexpr static pixel_format format() { return PF_RGBA8; }
    typedef vector<4, unorm< 8>> type;
};

template <>
struct pixel_traits<PF_RGB32F>
{
    VSNRAY_FUNC constexpr static pixel_format format() { return PF_RGB32F; }
    typedef vector<3, float> type;
};

template <>
struct pixel_traits<PF_RGBA32F>
{
    VSNRAY_FUNC constexpr static pixel_format format() { return PF_RGBA32F; }
    typedef vector<4, float> type;
};

} // visionaray

#endif // VSNRAY_PIXEL_TRAITS_H
