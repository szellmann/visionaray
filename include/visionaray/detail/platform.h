// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PLATFORM_H
#define VSNRAY_DETAIL_PLATFORM_H 1

#if defined(__APPLE__)
#define VSNRAY_OS_DARWIN 1
#elif defined(__linux__)
#define VSNRAY_OS_LINUX  1
#elif defined(_WIN32)
#define VSNRAY_OS_WIN32  1
#endif

#endif // VSNRAY_DETAIL_PLATFORM_H
