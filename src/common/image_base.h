// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_IMAGE_BASE_H
#define VSNRAY_COMMOM_IMAGE_BASE_H 1

#include <cstdint>
#include <utility>
#include <vector>

#include <visionaray/aligned_vector.h>

#include "file_base.h"
#include "pixel_format.h"

namespace visionaray
{

class image_base : public file_base
{
public:

    friend class image;

public:

    // Default constructor.
    image_base() = default;

    // Virtual default destructor.
    virtual ~image_base() = default;

    // Construct image from width, height, format, and data (data is copied).
    image_base(int width, int height, pixel_format format, uint8_t const* data);

    int width() const;
    int height() const;

    pixel_format format() const;

    uint8_t const* data() const;

protected:

    int width_;
    int height_;

    pixel_format format_ = PF_RGB8;

    aligned_vector<uint8_t> data_;

};

} // visionaray

#endif // VSNRAY_COMMOM_IMAGE_BASE_H
