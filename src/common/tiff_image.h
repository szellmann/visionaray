// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_TIFF_IMAGE_H
#define VSNRAY_COMMON_TIFF_IMAGE_H 1

#include <string>

#include "image_base.h"

namespace visionaray
{

class tiff_image : public image_base
{
public:

    bool load(std::string const& filename);

};

} // visionaray

#endif // VSNRAY_COMMON_TIFF_IMAGE_H
