// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_FILE_BASE_H
#define VSNRAY_COMMOM_FILE_BASE_H 1

#include <string>
#include <utility>
#include <vector>

#include <boost/any.hpp>

namespace visionaray
{

class file_base
{
public:

    virtual ~file_base() = default;

    using save_option  = std::pair<std::string, boost::any>;
    using save_options = std::vector<save_option>;

    virtual bool load(std::string const& filename);
    virtual bool save(std::string const& filename, save_options const& options);

};

} // visionaray

#endif // VSNRAY_COMMOM_FILE_BASE_H
