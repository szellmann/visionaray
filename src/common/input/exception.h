// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_EXCEPTION_H
#define VSNRAY_COMMON_INPUT_EXCEPTION_H 1

#include <visionaray/exception.h>

namespace visionaray
{

class invalid_key_modifier : public visionaray::exception
{
public:

    invalid_key_modifier(std::string const& what = "invalid key modifier")
        : exception(what)
    {
    }

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_EXCEPTION_H
