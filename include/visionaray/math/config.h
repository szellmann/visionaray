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
// Macros for CUDA host / device functions
//

#ifndef MATH_FUNC
#define MATH_FUNC __host__ __device__
#endif

#ifndef MATH_GPU_FUNC
#define MATH_GPU_FUNC __device__
#endif

#ifndef MATH_CPU_FUNC
#define MATH_CPU_FUNC __host__
#endif

#endif // VSNRAY_MATH_CONFIG_H
