// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VNSRAY_UTIL_H
#define VSNRAY_UTIL_H

#include <string>


namespace visionaray
{
namespace util
{

std::string backtrace();
std::string read_ascii(std::string const& filename);

} // util
} // visionaray

#endif // VSNRAY_UTIL_H


