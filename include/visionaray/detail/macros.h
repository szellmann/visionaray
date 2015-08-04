// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MACROS_H
#define VSNRAY_MACROS_H

#include "compiler.h"


/*! Align data on X-byte boundaries
 */
#if defined(_MSC_VER)
#define VSNRAY_ALIGN(X) __declspec(align(X))
#elif defined(__CUDACC__)
#define VSNRAY_ALIGN(X) __align__(X)
#else
#define VSNRAY_ALIGN(X) __attribute__((aligned(X)))
#endif


/*! Force function inlining
 */
#if VSNRAY_CXX_INTEL
#define VSNRAY_FORCE_INLINE __forceinline
#elif VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_FORCE_INLINE inline __attribute((always_inline))
#elif VSNRAY_CXX_MSVC
#define VSNRAY_FORCE_INLINE __forceinline
#else
#define VSNRAY_FORCE_INLINE inline
#endif


/*! mark CPU and GPU functions
 */
#ifdef __CUDACC__
#define VSNRAY_FUNC __device__ __host__
#define VSNRAY_GPU_FUNC __device__
#define VSNRAY_CPU_FUNC __host__
#else
#define VSNRAY_FUNC
#define VSNRAY_GPU_FUNC
#define VSNRAY_CPU_FUNC
#endif // __CUDACC__


/*! mark functions that are not expected to throw an exception
 */
#define VSNRAY_NOEXCEPT throw()


/*! Place in private section of class to disallow copying and assignment
 */
#define VSNRAY_NOT_COPYABLE(T)                                      \
  T(T const& rhs);                                                  \
  T& operator=(T const& rhs);


/*! mark variables thread-local
 */

#if VSNRAY_CXX_MSVC
#define VSNRAY_THREAD_LOCAL __declspec(thread)
#elif VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_THREAD_LOCAL __thread
#else
#define VSNRAY_THREAD_LOCAL
#endif

/*! Verbose way to say that a parameter is not used intentionally
 */
#define VSNRAY_UNUSED(x) ((void)(x))


/*! Mark code section unreachable
 */
#if VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_UNREACHABLE() __builtin_unreachable()
#elif VSNRAY_CXX_MSVC
#define VSNRAY_UNREACHABLE() __assume(0)
#else
#define VSNRAY_UNREACHABLE()
#endif


#endif // VSNRAY_MACROS_H
