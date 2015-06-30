// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/CmdLineUtil.h"
#include "Support/StringRef.h"

#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

using support::StringRef;

#ifdef _WIN32

static size_t EnumFiles(StringRef pattern, size_t wildpos, std::vector<std::string>& files)
{
    assert(wildpos != StringRef::npos);

    //
    // FIXME:
    // Use the Unicode version and convert strings to UTF-8?!?!
    //

    WIN32_FIND_DATAA info;
    HANDLE hFind = FindFirstFileExA(pattern.data(), FindExInfoStandard, &info, FindExSearchNameMatch, nullptr, 0);

    if (hFind == INVALID_HANDLE_VALUE)
        return 0;

    // Find first slash or colon before wildcard
    auto patternpos = pattern.find_last_of("\\/:", wildpos);
    auto size = files.size();

    do
    {
        StringRef arg = info.cFileName;

        if (arg == "." || arg == "..")
            continue;

        if (patternpos == StringRef::npos)
        {
            // Use current directory.
            files.push_back(arg.str());
        }
        else
        {
            // Path specified.
            // Prefix each filename with the path.
            files.push_back(pattern.substr(0, patternpos + 1).str() + arg.str());
        }
    }
    while (FindNextFileA(hFind, &info));

    FindClose(hFind);

    return files.size() - size;
}

#endif

void support::cl::expandWildcards(std::vector<std::string>& args)
{
#ifdef _WIN32

    std::vector<std::string> files;

    for (size_t i = 0; i != args.size(); )
    {
        files.clear();

        auto arg = StringRef(args[i]);
        auto wildpos = arg.find_first_of("*?");

        // If there is no '*' or '?'
        // or there are no matching files,
        // leave the pattern as is.
        if (wildpos == std::string::npos || EnumFiles(arg, wildpos, files) == 0)
        {
            ++i;
            continue;
        }

        // Sort the list.
        std::sort(files.begin(), files.end(),
            [](StringRef LHS, StringRef RHS) { return LHS < RHS; });

        // Replace the pattern with the first found file
        args[i] = std::move(files.front());

        // Insert the remaining files
        args.insert(args.begin() + (i + 1),
                    std::make_move_iterator(files.begin() + 1),
                    std::make_move_iterator(files.end()));

        i += files.size();

        assert(i <= args.size());
    }

#else

    static_cast<void>(args);

#endif
}
