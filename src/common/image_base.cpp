// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include "exception.h"
#include "image_base.h"

namespace visionaray
{

image_base::image_base(size_t width, size_t height, pixel_format format, uint8_t const* data)
    : width_(width)
    , height_(height)
    , format_(format)
{
    pixel_format_info info = map_pixel_format(format_);

    size_t len = width_ * height_ * info.size;
    data_.resize(len);

    std::copy(data, data + len, data_.begin());
}

bool image_base::load(std::string const& /*filename*/)
{
    throw visionaray::not_implemented_yet();
}

bool image_base::save(std::string const& /*filename*/, image_base::save_options const& /*options*/)
{
    throw visionaray::not_implemented_yet();
}

size_t image_base::width() const
{
    return width_;
}

size_t image_base::height() const
{
    return height_;
}

pixel_format image_base::format() const
{
    return format_;
}

uint8_t const* image_base::data() const
{
    return data_.data();
}

} // visionaray
