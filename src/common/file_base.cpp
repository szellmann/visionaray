// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdexcept>

#include "file_base.h"

namespace visionaray
{

bool file_base::load(std::string const& /*filename*/)
{
    throw std::runtime_error("Not implemented yet");
}

bool file_base::save(std::string const& /*filename*/, file_base::save_options const& /*options*/)
{
    throw std::runtime_error("Not implemented yet");
}

} // visionaray
