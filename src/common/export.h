// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_EXPORT_H
#define VSNRAY_COMMON_EXPORT_H 1

#include <visionaray/detail/compiler.h>

#ifndef VSNRAY_STATIC
#   ifdef visionaray_common_EXPORTS
#       define VSNRAY_COMMON_EXPORT VSNRAY_DLL_EXPORT
#   else
#       define VSNRAY_COMMON_EXPORT VSNRAY_DLL_IMPORT
#   endif
#else
#   define VSNRAY_COMMON_EXPORT
#endif

#endif // VSNRAY_COMMON_EXPORT_H
