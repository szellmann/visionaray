// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/exception.h>

#include "image_base.h"

namespace visionaray
{

bool image_base::load(std::string const& /*filename*/)
{
    throw visionaray::not_implemented_yet();
}

bool image_base::save(std::string const& /*filename*/)
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
