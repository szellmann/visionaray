// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_OBJ_LOADER_H
#define VSNRAY_COMMON_OBJ_LOADER_H

#include <string>

namespace visionaray
{

class model;

void load_obj(std::string const& filename, model& mod);

} // visionaray


#endif // VSNRAY_COMMON_OBJ_LOADER_H
