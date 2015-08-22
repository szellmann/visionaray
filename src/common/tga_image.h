// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_TGA_IMAGE_H
#define VSNRAY_COMMON_TGA_IMAGE_H 1

#include <cstddef>
#include <string>
#include <vector>

#include <visionaray/pixel_format.h>

namespace visionaray
{

class tga_image
{
public:

    tga_image(std::string const& filename);

    size_t width() const    { return width_; }
    size_t height() const   { return height_; }

    unsigned char const* data() const { return data_.data(); }

    pixel_format format() const { return format_; }

private:

    size_t width_;
    size_t height_;

    std::vector<unsigned char> data_;

    pixel_format format_;

};

} // visionaray

#endif // VSNRAY_COMMON_TGA_IMAGE_H
