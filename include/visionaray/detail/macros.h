// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_MACROS_H
#define VSNRAY_DETAIL_MACROS_H 1

#include "compiler.h"


//-------------------------------------------------------------------------------------------------
// VSNRAY_ALIGN(X)
// Align data on X-byte boundaries
//

#if defined(_MSC_VER)
#define VSNRAY_ALIGN(X) __declspec(align(X))
#elif defined(__CUDACC__)
#define VSNRAY_ALIGN(X) __align__(X)
#else
#define VSNRAY_ALIGN(X) __attribute__((aligned(X)))
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_WARN_UNUSED_RESULT
// Warn if result of function call is unused
//

#if VSNRAY_CXX_MSVC && (_MSC_VER >= 1700)
#define VSNRAY_WARN_UNUSED_RESULT _Check_return_
#else
#define VSNRAY_WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_FORCE_INLINE
// Force function inlining
//

#if VSNRAY_CXX_INTEL
#define VSNRAY_FORCE_INLINE __forceinline
#elif VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_FORCE_INLINE inline __attribute((always_inline))
#elif VSNRAY_CXX_MSVC
#define VSNRAY_FORCE_INLINE __forceinline
#else
#define VSNRAY_FORCE_INLINE inline
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_FUNC|VSNRAY_CPU_FUNC|VSNRAY_GPU_FUNC
// Mark CPU and GPU functions
//

#if defined(__CUDACC__) || defined(__HIPCC__)
#define VSNRAY_FUNC __device__ __host__
#define VSNRAY_GPU_FUNC __device__
#define VSNRAY_CPU_FUNC __host__
#else
#define VSNRAY_FUNC
#define VSNRAY_GPU_FUNC
#define VSNRAY_CPU_FUNC
#endif // __CUDACC__ || __HIPCC__


//-------------------------------------------------------------------------------------------------
// VSNRAY_CPU_MODE|VSNRAY_GPU_MODE
// Determine if code is compiled for the CPU or the GPU
//

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#define VSNRAY_CPU_MODE 0
#define VSNRAY_GPU_MODE 1
#else
#define VSNRAY_CPU_MODE 1
#define VSNRAY_GPU_MODE 0
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_CONSTEXPR|VSNRAY_DECL_CONSTEXPR
// Annotate expressions with C++11 constexpr if supported by the compiler
//

#ifdef VSNRAY_CXX_HAS_CONSTEXPR
#define VSNRAY_CONSTEXPR constexpr
#define VSNRAY_DECL_CONSTEXPR constexpr
#else
#define VSNRAY_CONSTEXPR const
#define VSNRAY_DECL_CONSTEXPR
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_NOEXCEPT
// Mark functions that are not expected to throw an exception (TODO: support C++-11 noexcept)
//

#define VSNRAY_NOEXCEPT throw()


//-------------------------------------------------------------------------------------------------
// VSNRAY_NOT_COPYABLE
// Place in private section of class to disallow copying and assignment
//

#define VSNRAY_NOT_COPYABLE(T)                                      \
  T(T const& rhs);                                                  \
  T& operator=(T const& rhs);


//-------------------------------------------------------------------------------------------------
// VSNRAY_THREAD_LOCAL
// Mark variables thread-local
//

#if VSNRAY_CXX_MSVC
#define VSNRAY_THREAD_LOCAL __declspec(thread)
#elif VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_THREAD_LOCAL __thread
#else
#define VSNRAY_THREAD_LOCAL
#endif


//-------------------------------------------------------------------------------------------------
// VSNRAY_UNUSED(...)
// Verbose way to say that a parameter is not used intentionally
//

#define VSNRAY_UNUSED(X) (void)(X)


//-------------------------------------------------------------------------------------------------
// VSNRAY_UNREACHABLE
// Mark code section unreachable
//

#if VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#define VSNRAY_UNREACHABLE __builtin_unreachable()
#elif VSNRAY_CXX_MSVC
#define VSNRAY_UNREACHABLE __assume(0)
#else
#define VSNRAY_UNREACHABLE
#endif


#endif // VSNRAY_DETAIL_MACROS_H
