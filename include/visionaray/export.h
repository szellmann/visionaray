// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_EXPORT_H
#define VSNRAY_EXPORT_H 1

#include "detail/compiler.h"

#ifndef VSNRAY_STATIC
#   ifdef visionaray_EXPORTS
#       define VSNRAY_EXPORT VSNRAY_DLL_EXPORT
#   else
#       define VSNRAY_EXPORT VSNRAY_DLL_IMPORT
#   endif
#else
#   define VSNRAY_EXPORT
#endif

#endif
