// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_COMPAT_H
#define VSNRAY_HCC_COMPAT_H 1

#include <cstddef>

//-------------------------------------------------------------------------------------------------
// Inject some functions into visionaray overload resolution set so that
// ANSI-C standard functions like assert() or memcpy() can be used w/o
// compiler errors
//

#if defined(__KALMAR_ACCELERATOR__) && __KALMAR_ACCELERATOR__ != 0
namespace visionaray
{

// Used in assert()
VSNRAY_FUNC inline void __assert_fail(...) {}

VSNRAY_FUNC inline void* memcpy(void* dest, void const* src, size_t n)
{
    char* d = reinterpret_cast<char*>(dest);
    char const* s = reinterpret_cast<char const*>(src);
    for (size_t i = 0; i < n; ++i)
    {
        *d++ = *s++;
    }
    return dest;
}

} // visionaray
#endif

#endif // VSNRAY_HCC_COMPAT_H
