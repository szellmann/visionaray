// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_COMPILER_H
#define VSNRAY_COMPILER_H


//
// Determine compiler
//

#if defined(__clang__)
#define VSNRAY_CXX_CLANG    (100 * __clang_major__ + __clang_minor__)
#elif defined(__INTEL_COMPILER) && defined(__GNUC__)
#define VSNRAY_CXX_INTEL    (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(__GNUC__)
#define VSNRAY_CXX_GCC      (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(_MSC_VER)
#define VSNRAY_CXX_MSVC     (_MSC_VER)
#endif

#ifndef VSNRAY_CXX_CLANG
#define VSNRAY_CXX_CLANG 0
#endif
#ifndef VSNRAY_CXX_INTEL
#define VSNRAY_CXX_INTEL 0
#endif
#ifndef VSNRAY_CXX_GCC
#define VSNRAY_CXX_GCC 0
#endif
#ifndef VSNRAY_CXX_MSVC
#define VSNRAY_CXX_MSVC 0
#endif

// MinGW on Windows?
#if defined(_WIN32) && (defined(__MINGW32__) || defined(__MINGW64__))
#define VSNRAY_CXX_MINGW VSNRAY_CXX_GCC
#else
#define VSNRAY_CXX_MINGW 0
#endif


//
// Determine available C++11 features
//

// VSNRAY_CXX_HAS_ALIAS_TEMPLATES
// VSNRAY_CXX_HAS_AUTO
// VSNRAY_CXX_HAS_CONSTEXPR
// VSNRAY_CXX_HAS_DECLTYPE
// VSNRAY_CXX_HAS_EXPLICIT_CONVERSIONS
// VSNRAY_CXX_HAS_GENERALIZED_INITIALIZERS
// VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
// VSNRAY_CXX_HAS_LAMBDAS
// VSNRAY_CXX_HAS_NULLPTR
// VSNRAY_CXX_HAS_OVERRIDE_CONTROL
// VSNRAY_CXX_HAS_RANGE_FOR
// VSNRAY_CXX_HAS_RVALUE_REFERENCES
// VSNRAY_CXX_HAS_STATIC_ASSERT
// VSNRAY_CXX_HAS_VARIADIC_TEMPLATES

#if VSNRAY_CXX_CLANG
#   if __has_feature(cxx_alias_templates)
#       define VSNRAY_CXX_HAS_ALIAS_TEMPLATES
#   endif
#   if __has_feature(cxx_auto_type)
#       define VSNRAY_CXX_HAS_AUTO
#   endif
#   if __has_feature(cxx_constexpr)
#       define VSNRAY_CXX_HAS_CONSTEXPR
#   endif
#   if __has_feature(cxx_decltype)
#       define VSNRAY_CXX_HAS_DECLTYPE
#   endif
#   if __has_feature(cxx_explicit_conversions)
#       define VSNRAY_CXX_HAS_EXPLICIT_CONVERSIONS
#   endif
#   if __has_feature(cxx_generalized_initializers)
#       define VSNRAY_CXX_HAS_GENERALIZED_INITIALIZERS
#   endif
#   if __has_feature(cxx_inheriting_constructors)
#       define VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
#   endif
#   if __has_feature(cxx_lambdas)
#       define VSNRAY_CXX_HAS_LAMBDAS
#   endif
#   if __has_feature(cxx_nullptr)
#       define VSNRAY_CXX_HAS_NULLPTR
#   endif
#   if __has_feature(cxx_override_control)
#       define VSNRAY_CXX_HAS_OVERRIDE_CONTROL
#   endif
#   if __has_feature(cxx_range_for)
#       define VSNRAY_CXX_HAS_RANGE_FOR
#   endif
#   if __has_feature(cxx_rvalue_references)
#       define VSNRAY_CXX_HAS_RVALUE_REFERENCES
#   endif
#   if __has_feature(cxx_static_assert)
#       define VSNRAY_CXX_HAS_STATIC_ASSERT
#   endif
#   if __has_feature(cxx_variadic_templates)
#       define VSNRAY_CXX_HAS_VARIADIC_TEMPLATES
#   endif
#elif VSNRAY_CXX_GCC
#   ifdef __GXX_EXPERIMENTAL_CXX0X__
#       if VSNRAY_CXX_GCC >= 403
#           define VSNRAY_CXX_HAS_DECLTYPE
#           define VSNRAY_CXX_HAS_RVALUE_REFERENCES
#           define VSNRAY_CXX_HAS_STATIC_ASSERT
#           define VSNRAY_CXX_HAS_VARIADIC_TEMPLATES
#       endif
#       if VSNRAY_CXX_GCC >= 404
#           define VSNRAY_CXX_HAS_AUTO
#           define VSNRAY_CXX_HAS_GENERALIZED_INITIALIZERS
#       endif
#       if VSNRAY_CXX_GCC >= 405
#           define VSNRAY_CXX_HAS_EXPLICIT_CONVERSIONS
#           define VSNRAY_CXX_HAS_LAMBDAS
#       endif
#       if VSNRAY_CXX_GCC >= 406
#           define VSNRAY_CXX_HAS_CONSTEXPR
#           define VSNRAY_CXX_HAS_NULLPTR
#           define VSNRAY_CXX_HAS_RANGE_FOR
#       endif
#       if VSNRAY_CXX_GCC >= 407
#           define VSNRAY_CXX_HAS_ALIAS_TEMPLATES
#           define VSNRAY_CXX_HAS_OVERRIDE_CONTROL
#       endif
#       if VSNRAY_CXX_GCC >= 408
#           define VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
#       endif
#   endif
#elif VSNRAY_CXX_MSVC
#   if VSNRAY_CXX_MSVC >= 1600 // Visual C++ 10.0 (2010)
#       define VSNRAY_CXX_HAS_AUTO
#       define VSNRAY_CXX_HAS_DECLTYPE
#       define VSNRAY_CXX_HAS_LAMBDAS
#       define VSNRAY_CXX_HAS_NULLPTR
#       define VSNRAY_CXX_HAS_RVALUE_REFERENCES
#       define VSNRAY_CXX_HAS_STATIC_ASSERT
#   endif
#   if VSNRAY_CXX_MSVC >= 1700 // Visual C++ 11.0 (2012)
#       define VSNRAY_CXX_HAS_OVERRIDE_CONTROL
#       define VSNRAY_CXX_HAS_RANGE_FOR
#   endif
#   if _MSC_FULL_VER == 170051025 // Visual C++ 12.0 November CTP
#       define VSNRAY_CXX_HAS_EXPLICIT_CONVERSIONS
#       define VSNRAY_CXX_HAS_GENERALIZED_INITIALIZERS
#       define VSNRAY_CXX_HAS_VARIADIC_TEMPLATES
#   endif
#endif


#ifdef VSNRAY_CXX_HAS_OVERRIDE_CONTROL
#define VSNRAY_OVERRIDE override
#define VSNRAY_FINAL final
#else
#if VSNRAY_CXX_MSVC
#define VSNRAY_OVERRIDE override
#else
#define VSNRAY_OVERRIDE
#endif
#define VSNRAY_FINAL
#endif


//
// Macros to work with compiler warnings/errors
//

// Use like: VSNRAY_CXX_MSVC_WARNING_DISABLE(4996)
// FIXME: Requires vc > ?
#if VSNRAY_CXX_MSVC
#   define VSNRAY_MSVC_WARNING_SUPPRESS(X) \
        __pragma(warning(suppress : X))
#   define VSNRAY_MSVC_WARNING_PUSH_LEVEL(X) \
        __pragma(warning(push, X))
#   define VSNRAY_MSVC_WARNING_PUSH() \
        __pragma(warning(push))
#   define VSNRAY_MSVC_WARNING_POP() \
        __pragma(warning(pop))
#   define VSNRAY_MSVC_WARNING_DEFAULT(X) \
        __pragma(warning(default : X))
#   define VSNRAY_MSVC_WARNING_DISABLE(X) \
        __pragma(warning(disable : X))
#   define VSNRAY_MSVC_WARNING_ERROR(X) \
        __pragma(warning(error : X))
#   define VSNRAY_MSVC_WARNING_PUSH_DISABLE(X) \
        __pragma(warning(push)) \
        __pragma(warning(disable : X))
#else
#   define VSNRAY_MSVC_WARNING_SUPPRESS(X)
#   define VSNRAY_MSVC_WARNING_PUSH_LEVEL(X)
#   define VSNRAY_MSVC_WARNING_PUSH()
#   define VSNRAY_MSVC_WARNING_POP()
#   define VSNRAY_MSVC_WARNING_DEFAULT(X)
#   define VSNRAY_MSVC_WARNING_DISABLE(X)
#   define VSNRAY_MSVC_WARNING_ERROR(X)
#   define SNRAYV_MSVC_WARNING_PUSH_DISABLE(X)
#endif


// Use like: VSNRAY_CXX_GCC_DIAGNOSTIC_IGNORE("-Wuninitialized")
// FIXME: Requires gcc > 4.6 or clang > ?.?
#if VSNRAY_CXX_CLANG || VSNRAY_CXX_GCC
#   define VSNRAY_GCC_DIAGNOSTIC_PUSH() \
        _Pragma("GCC diagnostic push")
#   define VSNRAY_GCC_DIAGNOSTIC_POP() \
        _Pragma("GCC diagnostic pop")
#   define VSNRAY_GCC_DIAGNOSTIC_IGNORE(X) \
        _Pragma("GCC diagnostic ignored \"" X "\"")
#   define VSNRAY_GCC_DIAGNOSTIC_WARNING(X) \
        _Pragma("GCC diagnostic warning \"" X "\"")
#   define VSNRAY_GCC_DIAGNOSTIC_ERROR(X) \
        _Pragma("GCC diagnostic error \"" X "\"")
#else
#   define VSNRAY_GCC_DIAGNOSTIC_PUSH()
#   define VSNRAY_GCC_DIAGNOSTIC_POP()
#   define VSNRAY_GCC_DIAGNOSTIC_IGNORE(X)
#   define VSNRAY_GCC_DIAGNOSTIC_WARNING(X)
#   define VSNRAY_GCC_DIAGNOSTIC_ERROR(X)
#endif


// Macro to mark classes and functions as deprecated. Usage:
//
// struct VSNRAY_DEPRECATED X {};
// struct X { void fun(); void VSNRAY_DEPRECATED fun(int); }
//
// void VSNRAY_DEPRECATED fun();
//
#if VSNRAY_CXX_MSVC
#   define VSNRAY_DEPRECATED __declspec(deprecated)
#elif VSNRAY_CXX_GCC || VSNRAY_CXX_CLANG
#   define VSNRAY_DEPRECATED __attribute__((deprecated))
#else
#   define VSNRAY_DEPRECATED
#endif


#if VSNRAY_CXX_MSVC
#   define VSNRAY_DLL_EXPORT __declspec(dllexport)
#   define VSNRAY_DLL_IMPORT __declspec(dllimport)
#elif VSNRAY_CXX_CLANG || VSNRAY_CXX_GCC
#   define VSNRAY_DLL_EXPORT __attribute__((visibility("default")))
#   define VSNRAY_DLL_IMPORT __attribute__((visibility("default")))
#else
#   define VSNRAY_DLL_EXPORT
#   define VSNRAY_DLL_IMPORT
#endif


#endif
