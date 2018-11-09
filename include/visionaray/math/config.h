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
#ifdef __CUDACC__
#define MATH_FUNC __host__ __device__
#else
#define MATH_FUNC
#endif
#endif

#ifndef MATH_GPU_FUNC
#ifdef __CUDACC__
#define MATH_GPU_FUNC __device__
#else
#define MATH_GPU_FUNC
#endif
#endif

#ifndef MATH_CPU_FUNC
#ifdef __CUDACC__
#define MATH_CPU_FUNC __host__
#else
#define MATH_CPU_FUNC
#endif
#endif

#endif // VSNRAY_MATH_CONFIG_H
