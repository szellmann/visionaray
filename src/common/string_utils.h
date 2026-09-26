// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_STRING_UTILS_H
#define VSNRAY_COMMON_STRING_UTILS_H 1

#include <cctype>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace visionaray
{

inline std::string trim(std::string str, std::string ws = " \t")
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

inline std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}

inline size_t count_whitespaces(std::string str)
{
    return std::count_if(
            str.begin(),
            str.end(),
            [](unsigned char c) { return std::isspace(c); }
            );
}

inline std::string tolower(std::string str)
{
    std::transform(
            str.begin(),
            str.end(),
            str.begin(),
            [](unsigned char c) { return std::tolower(c); }
            );

    return str;
}

} // visionaray

#endif // VSNRAY_COMMON_STRING_UTILS_H
