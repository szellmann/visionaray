// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_IMAGE_BASE_H
#define VSNRAY_COMMOM_IMAGE_BASE_H 1

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/any.hpp>

#include <visionaray/aligned_vector.h>
#include <visionaray/pixel_format.h>

namespace visionaray
{

class image_base
{
public:

    friend class image;

    using save_option  = std::pair<std::string, boost::any>;
    using save_options = std::vector<save_option>;

public:

    // Default constructor.
    image_base() = default;

    // Construct image from width, height, format, and data (data is copied).
    image_base(size_t width, size_t height, pixel_format format, uint8_t const* data);

    virtual bool load(std::string const& filename);
    virtual bool save(std::string const& filename, save_options const& options);

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
