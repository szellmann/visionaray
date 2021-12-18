// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_OBJ_LOADER_H
#define VSNRAY_COMMON_OBJ_LOADER_H 1

#include <string>
#include <vector>

#include "export.h"

namespace visionaray
{

class model;

VSNRAY_COMMON_EXPORT void load_obj(std::string const& filename, model& mod);
VSNRAY_COMMON_EXPORT void load_obj(std::vector<std::string> const& filenames, model& mod);

} // visionaray

#endif // VSNRAY_COMMON_OBJ_LOADER_H
