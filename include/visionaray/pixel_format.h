// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PIXEL_FORMAT_H
#define VSNRAY_PIXEL_FOMRAT_H 1

#include <type_traits>

namespace visionaray
{


enum pixel_format
{
    PF_UNSPECIFIED,

    // pixel formats for color buffers and images

    PF_R8,
    PF_RG8,
    PF_RGB8,
    PF_RGBA8,
    PF_R16F,
    PF_RG16F,
    PF_RGB16F,
    PF_RGBA16F,
    PF_R32F,
    PF_RG32F,
    PF_RGB32F,
    PF_RGBA32F,
    PF_R16I,
    PF_RG16I,
    PF_RGB16I,
    PF_RGBA16I,
    PF_R32I,
    PF_RG32I,
    PF_RGB32I,
    PF_RGBA32I,
    PF_R16UI,
    PF_RG16UI,
    PF_RGB16UI,
    PF_RGBA16UI,
    PF_R32UI,
    PF_RG32UI,
    PF_RGB32UI,
    PF_RGBA32UI,

    PF_BGR8,
    PF_BGRA8,

    PF_RGB10_A2,

    PF_R11F_G11F_B10F,

    // pixel formats for depth and stencil buffers

    PF_DEPTH16,
    PF_DEPTH24,
    PF_DEPTH32,
    PF_DEPTH32F,
    PF_DEPTH24_STENCIL8,
    PF_DEPTH32F_STENCIL8,

    PF_LUMINANCE8,
    PF_LUMINANCE16,
    PF_LUMINANCE32F,

    PF_COUNT // last
};


struct pixel_format_info
{
    unsigned internal_format;
    unsigned format;
    unsigned type;
    unsigned components;
    unsigned size;
};


//-------------------------------------------------------------------------------------------------
// pixel_format_constant e.g. for use with tag dispatch
//

template <pixel_format PF>
using pixel_format_constant = std::integral_constant<pixel_format, PF>;


pixel_format      map_gl_format(unsigned format, unsigned type, unsigned size);
pixel_format_info map_pixel_format(pixel_format format);

} // visionaray

#endif // VSNRAY_PIXEL_FORMAT_H
