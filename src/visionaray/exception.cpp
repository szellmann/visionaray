// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "exception.h"
#include "util.h"


using visionaray::exception;


exception::exception(std::string const& what)
    : what_(what)
    , where_(visionaray::util::backtrace())
{
}


char const* exception::what() const VSNRAY_NOEXCEPT
{
    return what_.c_str();
}


char const* exception::where() const VSNRAY_NOEXCEPT
{
    return where_.c_str();
}


