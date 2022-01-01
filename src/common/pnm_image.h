// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_PNM_IMAGE_H
#define VSNRAY_COMMON_PNM_IMAGE_H 1

#include <string>

#include "export.h"
#include "image_base.h"

namespace visionaray
{

class pnm_image : public image_base
{
public:

    // Default constructor.
    VSNRAY_COMMON_EXPORT pnm_image() = default;

    // Construct image from width, height, format, and data (data is copied).
    VSNRAY_COMMON_EXPORT pnm_image(int width, int height, pixel_format format, uint8_t const* data);

    VSNRAY_COMMON_EXPORT bool load(std::string const& filename);

    // Save pnm image. Options: { "binary", <bool> }
    VSNRAY_COMMON_EXPORT bool save(std::string const& filename, save_options const& options);

};

} // visionaray

#endif // VSNRAY_COMMON_PNM_IMAGE_H
