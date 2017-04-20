// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

//--------------------------------------------------------------------------------------------------
// Detect architecture
//

// Base arch (same for 32-bit and 64-bit)
#define VSNRAY_BASE_ARCH_UNKNOWN  9999
#define VSNRAY_BASE_ARCH_X86         0
#define VSNRAY_BASE_ARCH_ARM      1000

#define VSNRAY_ARCH_UNKNOWN          0
#define VSNRAY_ARCH_X86             10
#define VSNRAY_ARCH_X86_64          11
#define VSNRAY_ARCH_ARM             20
#define VSNRAY_ARCH_ARM64           21

#if defined(_M_X64) || defined(_M_AMD64) || defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64)
#define VSNRAY_BASE_ARCH VSNRAY_BASE_ARCH_X86
#define VSNRAY_ARCH VSNRAY_ARCH_X86_64
#elif defined(__arm__) || defined(__arm) || defined(_ARM) || defined(_M_ARM)
#define VSNRAY_BASE_ARCH VSNRAY_BASE_ARCH_ARM
#define VSNRAY_ARCH VSNRAY_ARCH_ARM
#elif defined(__aarch64__)
#define VSNRAY_BASE_ARCH VSNRAY_BASE_ARCH_ARM
#define VSNRAY_ARCH VSNRAY_ARCH_ARM64
#else
#define VSNRAY_BASE_ARCH VSNRAY_BASE_ARCH_UNKNOWN
#define VSNRAY_ARCH VSNRAY_ARCH_UNKNOWN
#endif

//--------------------------------------------------------------------------------------------------
// Detect instruction set
//

// x86 [0-1000)
#define VSNRAY_SIMD_ISA_SSE       10
#define VSNRAY_SIMD_ISA_SSE2      20
#define VSNRAY_SIMD_ISA_SSE3      30
#define VSNRAY_SIMD_ISA_SSSE3     31
#define VSNRAY_SIMD_ISA_SSE4_1    41
#define VSNRAY_SIMD_ISA_SSE4_2    42
#define VSNRAY_SIMD_ISA_AVX       50
#define VSNRAY_SIMD_ISA_AVX2      60
#define VSNRAY_SIMD_ISA_AVX512F   70

// ARM [1000-2000)
#define VSNRAY_SIMD_ISA_NEON    1010
#define VSNRAY_SIMD_ISA_NEON_FP 1020

#ifndef VSNRAY_SIMD_ISA__
#if defined(__AVX512F__)                            && !defined(__CUDACC__) // nvcc does not support AVX intrinsics
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_AVX512F
#elif defined(__AVX2__)                             && !defined(__CUDACC__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_AVX2
#elif defined(__AVX__)                              && !defined(__CUDACC__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_AVX
#elif defined(__SSE4_2__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_SSE4_2
#elif defined(__SSE4_1__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_SSE4_1
#elif defined(__SSSE3__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_SSSE3
#elif defined(__SSE3__)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_SSE3
#elif defined(__SSE2__) || VSNRAY_ARCH == VSNRAY_ARCH_X86_64 // SSE2 is always available on 64-bit Intel compatible platforms
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_SSE2
#elif defined(__ARM_NEON) && defined(__ARM_NEON_FP)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_NEON_FP
#elif defined(__ARM_NEON)
#define VSNRAY_SIMD_ISA__ VSNRAY_SIMD_ISA_NEON
#else
#define VSNRAY_SIMD_ISA__ 0
#endif
#endif

// Undef VSNRAY_SIMD_ISA__ when compiling device code with hcc
#if defined(__KALMAR_ACCELERATOR__)
#ifdef VSNRAY_SIMD_ISA__
#undef VSNRAY_SIMD_ISA__
#endif
#define VSNRAY_SIMD_ISA__ 0
#endif

// Intel Short Vector Math Library available?
#ifndef VSNRAY_SIMD_HAS_SVML
#if defined(__INTEL_COMPILER)
#define VSNRAY_SIMD_HAS_SVML 1
#endif
#endif

//-------------------------------------------------------------------------------------------------
// Macros to identify SIMD isa availability
//

#define VSNRAY_NO_SIMD_ISA                                                      \
    VSNRAY_SIMD_ISA__ == 0

#define VSNRAY_SIMD_ISA_EQ(ISA)                                                 \
    ISA - VSNRAY_BASE_ARCH >= 0 &&                                              \
    ISA - VSNRAY_BASE_ARCH < 1000 &&                                            \
    VSNRAY_SIMD_ISA__ == ISA

#define VSNRAY_SIMD_ISA_GE(ISA)                                                 \
    ISA - VSNRAY_BASE_ARCH >= 0 &&                                              \
    ISA - VSNRAY_BASE_ARCH < 1000 &&                                            \
    VSNRAY_SIMD_ISA__ >= ISA

//--------------------------------------------------------------------------------------------------
// SIMD intrinsic #include's
//

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE2)
#include <emmintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE3)
#include <pmmintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSSE3)
#include <tmmintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_1)
#include <smmintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE4_2)
#include <nmmintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
#include <immintrin.h>
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_NEON)
#include <arm_neon.h>
#endif
