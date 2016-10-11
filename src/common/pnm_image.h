// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VNSRAY_COMMON_PNM_IMAGE_H
#define VNSRAY_COMMON_PNM_IMAGE_H 1

#include <string>

#include "image_base.h"

namespace visionaray
{

class pnm_image : public image_base
{
public:

    bool load(std::string const& filename);

};

} // visionaray

#endif // VNSRAY_COMMON_PNM_IMAGE_H
