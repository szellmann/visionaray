// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INIFILE_H
#define VSNRAY_COMMON_INIFILE_H 1

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Very basic ini file parser
//

class inifile
{
public:

    using key_type = std::string;
    struct value_type
    {
        std::string section;
        std::string value;
    };

    enum error_code
    {
        Ok, ParseError, NonExistent, MultipleEntries
    };

public:

    inifile(std::string filename);

    bool good() const;

    error_code get_bool(std::string key, bool& value);

    error_code get_int8(std::string key, int8_t& value);
    error_code get_int16(std::string key, int16_t& value);
    error_code get_int32(std::string key, int32_t& value);
    error_code get_int64(std::string key, int64_t& value);

    error_code get_uint8(std::string key, uint8_t& value);
    error_code get_uint16(std::string key, uint16_t& value);
    error_code get_uint32(std::string key, uint32_t& value);
    error_code get_uint64(std::string key, uint64_t& value);

    error_code get_float(std::string key, float& value);
    error_code get_double(std::string key, double& value);
    error_code get_long_double(std::string key, long double& value);

    error_code get_string(std::string key, std::string& value, bool remove_quotes = false);

    error_code get_vec3i(std::string key, int32_t& x, int32_t& y, int32_t& z);
    error_code get_vec3ui(std::string key, uint32_t& x, uint32_t& y, uint32_t& z);
    error_code get_vec3f(std::string key, float& x, float& y, float& z);
    error_code get_vec3d(std::string key, double& x, double& y, double& z);

private:

    std::ifstream file_;
    std::multimap<key_type, value_type> entries_;

    bool good_ = false;

};

} // visionaray

#endif // VSNRAY_COMMON_INIFILE_H
