// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_CONFIG_H
#define VSNRAY_MATH_CONFIG_H 1

//--------------------------------------------------------------------------------------------------
// Project specific default namespace
//

#ifndef MATH_NAMESPACE
#define MATH_NAMESPACE visionaray
#endif


//-------------------------------------------------------------------------------------------------
// Project specific macros for host or device functions (or both)
//

#include <visionaray/detail/macros.h>

#ifndef MATH_FUNC
#define MATH_FUNC VSNRAY_FUNC
#endif

#ifndef MATH_GPU_FUNC
#define MATH_GPU_FUNC VSNRAY_GPU_FUNC
#endif

#ifndef MATH_CPU_FUNC
#define MATH_CPU_FUNC VSNRAY_CPU_FUNC
#endif

#endif // VSNRAY_MATH_CONFIG_H


