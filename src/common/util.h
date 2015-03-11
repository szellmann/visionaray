// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once 

#ifndef VSNRAY_COMMON_UTIL_H
#define VSNRAY_COMMON_UTIL_H

#include <visionaray/detail/macros.h>

#if defined(VSNRAY_OS_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace visionaray
{

inline unsigned get_num_processors()
{
#if defined(VSNRAY_OS_WIN32)

    SYSTEM_INFO sysinfo;
    GetNativeSystemInfo(&sysinfo);
    return static_cast<unsigned>(sysinfo.dwNumberOfProcessors);

#else

    return static_cast<unsigned>(sysconf(_SC_NPROCESSORS_ONLN));

#endif
}

}

#endif
