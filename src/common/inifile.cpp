// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <vector>

#include "inifile.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Helpers
//

std::string trim(std::string str, std::string ws = " \t")
{
    // Remove leading whitespace
    auto first = str.find_first_not_of(ws);

    // Only whitespace found
    if (first == std::string::npos)
    {
        return "";
    }

    // Remove trailing whitespace
    auto last = str.find_last_not_of(ws);

    // No whitespace found
    if (last == std::string::npos)
    {
        last = str.size() - 1;
    }

    // Skip if empty
    if (first > last)
    {
        return "";
    }

    // Trim
    return str.substr(first, last - first + 1);
}

std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}

size_t count_whitespaces(std::string str)
{
    return std::count_if(
            str.begin(),
            str.end(),
            [](unsigned char c) { return std::isspace(c); }
            );
}

std::string tolower(std::string str)
{
    std::transform(
            str.begin(),
            str.end(),
            str.begin(),
            [](unsigned char c) { return std::tolower(c); }
            );

    return str;
}

template <typename T>
inline bool as_T(std::string in, T& out)
{
    std::istringstream stream(in);
    return static_cast<bool>(stream >> out);
}

template <typename Map>
inline inifile::error_code get_string(Map const& map, std::string key, std::string& value)
{
    if (map.count(key) > 1)
    {
        return inifile::MultipleEntries;
    }

    auto it = map.find(key);

    if (it == map.end())
    {
        return inifile::NonExistent;
    }

    value = trim(it->second.value);

    return inifile::Ok;
}

template <typename Map, typename T>
inline inifile::error_code get_T(Map const& map, std::string key, T& value)
{
    std::string str = "";
    inifile::error_code err = get_string(map, key, str);
    if (err != inifile::Ok)
    {
        return err;
    }

    if (!as_T(str, value))
    {
        return inifile::ParseError;
    }

    return inifile::Ok;
}

template <typename Map, typename T>
inline inifile::error_code get_T3(Map const& map, std::string key, T& x, T& y, T& z)
{
    std::string str = "";
    inifile::error_code err = get_string(map, key, str);
    if (err != inifile::Ok)
    {
        return err;
    }

    // Also remove enclosing parentheses
    str = trim(str, "()");
    str = trim(str, "[]");
    str = trim(str, "{}");

    std::string delims = " ,;|";

    for (auto c : delims)
    {
        auto tokens = string_split(str, c);

        if (tokens.size() >= 3)
        {
            std::string xyz[3];
            int c = 0;
            for (auto t : tokens)
            {
                if (count_whitespaces(t) < t.size())
                {
                    xyz[c++] = t;
                }
            }

            if (c == 3 && as_T(xyz[0], x) && as_T(xyz[1], y) && as_T(xyz[2], z))
            {
                return inifile::Ok;
            }
        }
    }

    return inifile::ParseError;
}


//-------------------------------------------------------------------------------------------------
// Interface
//

inifile::inifile(std::string filename)
    : file_(filename)
{
    if (!file_.good())
    {
        return;
    }

    std::string section = "";

    for (std::string line; std::getline(file_, line); )
    {
        line = trim(line);
        if (line.empty())
        {
            continue;
        }

        // Skip comments
        if (line[0] == ';' || line[0] == '#')
        {
            continue;
        }

        // Section
        if (line[0] == '[' && line[line.size() - 1] == ']')
        {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }

        // Parse key/value pairs
        auto p = string_split(line, '=');

        if (p.size() != 2)
        {
            // TODO: error handling
            continue;
        }

        std::string key = tolower(trim(p[0]));
        std::string value = tolower(trim(p[1]));

        entries_.emplace(std::make_pair(key, value_type{ section, value }));
    }

    good_ = true;
}

bool inifile::good() const
{
    return good_;
}

inifile::error_code inifile::get_bool(std::string key, bool& value)
{
    std::string str = "";
    inifile::error_code err = visionaray::get_string(entries_, key, str);
    if (err != inifile::Ok)
    {
        return err;
    }

    str = tolower(str);

    if (str == "1" || str == "true" || str == "on")
    {
        value = true;
        return Ok;
    }
    else if (str == "0" || str == "false" || str == "off")
    {
        value = false;
        return Ok;
    }
    else
    {
        return ParseError;
    }
}

inifile::error_code inifile::get_int8(std::string key, int8_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_int16(std::string key, int16_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_int32(std::string key, int32_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_int64(std::string key, int64_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_uint8(std::string key, uint8_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_uint16(std::string key, uint16_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_uint32(std::string key, uint32_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_uint64(std::string key, uint64_t& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_float(std::string key, float& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_double(std::string key, double& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_long_double(std::string key, long double& value)
{
    return get_T(entries_, key, value);
}

inifile::error_code inifile::get_string(std::string key, std::string& value, bool remove_quotes)
{
    error_code err = visionaray::get_string(entries_, key, value);

    if (err == Ok && remove_quotes)
    {
        // TODO: check that we removed *2* (and both the same) quotation marks
        value = trim(value, "'");
        value = trim(value, "\"");
    }

    return err;
}

inifile::error_code inifile::get_vec3i(std::string key, int32_t& x, int32_t& y, int32_t& z)
{
    return get_T3(entries_, key, x, y, z);
}

inifile::error_code inifile::get_vec3ui(std::string key, uint32_t& x, uint32_t& y, uint32_t& z)
{
    return get_T3(entries_, key, x, y, z);
}

inifile::error_code inifile::get_vec3f(std::string key, float& x, float& y, float& z)
{
    return get_T3(entries_, key, x, y, z);
}

inifile::error_code inifile::get_vec3d(std::string key, double& x, double& y, double& z)
{
    return get_T3(entries_, key, x, y, z);
}

} // visionaray
