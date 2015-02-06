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
    /* No pixel_traits::{format|type} causes a compiler error */
};

template <>
struct pixel_traits<PF_UNSPECIFIED>
{
    static const pixel_format format = PF_UNSPECIFIED;
    typedef struct {} type;
};

template <>
struct pixel_traits<PF_RGB8>
{
    static const pixel_format format = PF_RGB8;
    typedef vector<3, unorm< 8>> type;
};

template <>
struct pixel_traits<PF_RGBA8>
{
    static const pixel_format format = PF_RGBA8;
    typedef vector<4, unorm< 8>> type;
};

template <>
struct pixel_traits<PF_RGB32F>
{
    static const pixel_format format = PF_RGB32F;
    typedef vector<3, float> type;
};

template <>
struct pixel_traits<PF_RGBA32F>
{
    static const pixel_format format = PF_RGBA32F;
    typedef vector<4, float> type;
};

} // visionaray

#endif // VSNRAY_PIXEL_TRAITS_H
