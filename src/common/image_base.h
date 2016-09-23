// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_IMAGE_BASE_H
#define VSNRAY_COMMOM_IMAGE_BASE_H 1

#include <cstddef>
#include <cstdint>
#include <string>

#include <visionaray/aligned_vector.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{

class image_base
{
public:

    friend class image;

public:

    virtual bool load(std::string const& filename);
    virtual bool save(std::string const& filename);

    size_t width() const;
    size_t height() const;

    pixel_format format() const;

    uint8_t const* data() const;

protected:

    size_t width_;
    size_t height_;

    pixel_format format_ = PF_RGB8;

    aligned_vector<uint8_t> data_;

};

} // visionaray

#endif // VSNRAY_COMMOM_IMAGE_BASE_H
