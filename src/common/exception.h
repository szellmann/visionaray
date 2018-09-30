// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_EXCEPTION_H
#define VSNRAY_COMMON_EXCEPTION_H 1

#include <stdexcept>

#include <visionaray/detail/macros.h>
#include <visionaray/export.h>

namespace visionaray
{

class VSNRAY_EXPORT not_implemented_yet : public std::logic_error
{
public:
    not_implemented_yet()
        : std::logic_error("Not implemented yet")
    {
    }
};

} // visionaray

#endif // VSNRAY_COMMON_EXCEPTION_H
