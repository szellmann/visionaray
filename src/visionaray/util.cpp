// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN) || defined(VSNRAY_OS_LINUX)
#include <execinfo.h>
#endif

#include "util.h"


namespace visionaray
{
namespace util
{

std::string backtrace()
{
#if defined(VSNRAY_OS_DARWIN) || defined(VSNRAY_OS_LINUX)
    static const int max_frames = 16;

    void* buffer[max_frames] = { 0 };
    int cnt = ::backtrace(buffer, max_frames);

    char** symbols = backtrace_symbols(buffer, cnt);

    if (symbols)
    {
        std::stringstream str;
        for (int n = 1; n < cnt; ++n) // skip the 1st entry (address of this function)
        {
            str << symbols[n] << std::endl;
        }
        free(symbols);
        return str.str();
    }
    return std::string();
#endif
}


std::string read_ascii(std::string const& filename)
{
    std::ifstream file(filename.c_str());

    if (!file.is_open())
    {
        return "";
    }

    return std::string(
        std::istreambuf_iterator<char>(file.rdbuf()),
        std::istreambuf_iterator<char>());
}

} // util
} // visionaray


