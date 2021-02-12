// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_PNG_IMAGE_H
#define VSNRAY_COMMON_PNG_IMAGE_H 1

#include <cstdint>
#include <string>

#include "image_base.h"

namespace visionaray
{

class png_image : public image_base
{
public:

    // Default constructor.
    png_image() = default;

    // Construct image from width, height, format, and data (data is copied).
    png_image(int width, int height, pixel_format format, uint8_t const* data);

    bool load(std::string const& filename);

    // Save png image. Options: { tba. }
    bool save(std::string const& filename, save_options const& options);

};

} // visionaray

#endif // VSNRAY_COMMON_PNG_IMAGE_H
