// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_VSNRAY_LOADER_H
#define VSNRAY_COMMON_VSNRAY_LOADER_H 1

#include <string>
#include <vector>

namespace visionaray
{

class model;

void load_vsnray(std::string const& filename, model& mod);
void load_vsnray(std::vector<std::string> const& filenames, model& mod);

} // visionaray

#endif // VSNRAY_COMMON_VSNRAY_LOADER_H
