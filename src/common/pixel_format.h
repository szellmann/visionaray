// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_PIXEL_FORMAT_H
#define VSNRAY_COMMON_PIXEL_FOMRAT_H 1

#include <visionaray/pixel_format.h>

#include <type_traits>

namespace visionaray
{

struct pixel_format_info
{
    unsigned internal_format;
    unsigned format;
    unsigned type;
    unsigned components;
    unsigned size;
};

pixel_format      map_gl_format(unsigned format, unsigned type, unsigned size);
pixel_format_info map_pixel_format(pixel_format format);

} // visionaray

#endif // VSNRAY_COMMON_PIXEL_FORMAT_H
