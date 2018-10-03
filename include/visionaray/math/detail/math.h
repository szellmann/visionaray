// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_DETAIL_MATH_H
#define VSNRAY_MATH_DETAIL_MATH_H 1

#ifdef __HCC__
#include <hcc/hc_math.hpp>
#else
#include <cmath>
#endif
#include <type_traits>

#include "../config.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// Import required math functions from the standard library.
// Enable ADL!
//

#if defined(__CUDA_ARCH__)
using ::abs;
using ::acos;
using ::asin;
using ::atan;
using ::atan2;
using ::ceil;
using ::copysign;
using ::cos;
using ::exp;
using ::floor;
using ::isfinite;
using ::isinf;
using ::isnan;
using ::log;
using ::log2;
using ::log10;
using ::log1p;
using ::pow;
using ::round;
using ::sin;
using ::sqrt;
using ::tan;
#elif defined(__KALMAR_ACCELERATOR__) && __KALMAR_ACCELERATOR__
MATH_FUNC inline float abs(float x) { return hc::precise_math::fabsf(x); }
MATH_FUNC inline float acos(float x) { return hc::precise_math::acosf(x); }
MATH_FUNC inline float asin(float x) { return hc::precise_math::asinf(x); }
MATH_FUNC inline float atan(float x) { return hc::precise_math::atanf(x); }
MATH_FUNC inline float atan2(float y, float x) { return hc::precise_math::atanf(y, x); }
MATH_FUNC inline float ceil(float x) { return hc::precise_math::ceilf(x); }
MATH_FUNC inline float copysign(float x, float y) { return hc::precise_math::copysignf(x, y); }
MATH_FUNC inline float cos(float x) { return hc::precise_math::cosf(x); }
MATH_FUNC inline float exp(float x) { return hc::precise_math::expf(x); }
MATH_FUNC inline float floor(float x) { return hc::precise_math::floorf(x); }
MATH_FUNC inline int isfinite(float x) { return hc::precise_math::isfinite(x); }
MATH_FUNC inline int isinf(float x) { return hc::precise_math::isinf(x); }
MATH_FUNC inline int isnan(float x) { return hc::precise_math::isnan(x); }
MATH_FUNC inline float log(float x) { return hc::precise_math::logf(x); }
MATH_FUNC inline float log2(float x) { return hc::precise_math::log2f(x); }
MATH_FUNC inline float log10(float x) { return hc::precise_math::log10f(x); }
MATH_FUNC inline float log1p(float x) { return hc::precise_math::log1pf(x); }
MATH_FUNC inline float pow(float x, float y) { return hc::precise_math::powf(x, y); }
MATH_FUNC inline float round(float x) { return hc::precise_math::roundf(x); }
MATH_FUNC inline float sin(float x) { return hc::precise_math::sinf(x); }
MATH_FUNC inline float sqrt(float x) { return hc::precise_math::sqrtf(x); }
MATH_FUNC inline float tan(float x) { return hc::precise_math::tanf(x); }
#else
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::copysign;
using std::cos;
using std::exp;
using std::floor;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::log;
using std::log2;
using std::log10;
using std::log1p;
using std::pow;
using std::round;
using std::sin;
using std::sqrt;
using std::tan;
#endif

template <typename T>
MATH_FUNC
inline T min(T const& x, T const& y)
{
    return x < y ? x : y;
}

template <typename T>
MATH_FUNC
inline T max(T const& x, T const& y)
{
    return x < y ? y : x;
}


//-------------------------------------------------------------------------------------------------
// Conversion functions, more useful when used with SIMD types
//

MATH_FUNC
inline int reinterpret_as_int(float a)
{
    // Prefer union over reinterpret_cast for type-punning
    // for compilers with strict-aliasing rules
    union helper
    {
        float a;
        int i;
    };
    helper h;
    h.a = a;
    return h.i;
}

MATH_FUNC
inline float reinterpret_as_float(int a)
{
    // Prefer union over reinterpret_cast for type-punning
    // for compilers with strict-aliasing rules
    union helper
    {
        int a;
        float f;
    };
    helper h;
    h.a = a;
    return h.f;
}

MATH_FUNC
inline float convert_to_float(int a)
{
    return static_cast<float>(a);
}

MATH_FUNC
inline int convert_to_int(float a)
{
    return static_cast<int>(a);
}


//-------------------------------------------------------------------------------------------------
// Extended versions of min/max
//

template <typename T>
MATH_FUNC
inline T min(T const& x, T const& y, T const& z)
{
    return min( min(x, y), z );
}

template <typename T>
MATH_FUNC
inline T max(T const& x, T const& y, T const& z)
{
    return max( max(x, y), z );
}

template <typename T>
MATH_FUNC
inline T min_max(T const& x, T const& y, T const& z)
{
    return max( min(x, y), z );
}

template <typename T>
MATH_FUNC
inline T max_min(T const& x, T const& y, T const& z)
{
    return min( max(x, y), z );
}

#ifdef __CUDA_ARCH__

MATH_GPU_FUNC
inline int min(int x, int y, int z)
{
    int result;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
}

MATH_GPU_FUNC
inline int max(int x, int y, int z)
{
    int result;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
}

MATH_GPU_FUNC
inline int min_max(int x, int y, int z)
{
    int result;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
}

MATH_GPU_FUNC
inline int max_min(int x, int y, int z)
{
    int result;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
}

MATH_GPU_FUNC
inline float min(float x, float y, float z)
{
    return __int_as_float( min(__float_as_int(x), __float_as_int(y), __float_as_int(z)) );
}

MATH_GPU_FUNC
inline float max(float x, float y, float z)
{
    return __int_as_float( max(__float_as_int(x), __float_as_int(y), __float_as_int(z)) );
}

MATH_GPU_FUNC
inline float min_max(float x, float y, float z)
{
    return __int_as_float( min_max(__float_as_int(x), __float_as_int(y), __float_as_int(z)) );
}

MATH_GPU_FUNC
inline float max_min(float x, float y, float z)
{
    return __int_as_float( max_min(__float_as_int(x), __float_as_int(y), __float_as_int(z)) );
}

#endif


//-------------------------------------------------------------------------------------------------
// Round (a) up to the nearest multiple of (b), then divide by (b)
//

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type div_up(T a, T b)
{
    return (a + b - 1) / b;
}

//-------------------------------------------------------------------------------------------------
// Round (a) up to the nearest multiple of (b)
//

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type round_up(T a, T b)
{
    return div_up(a, b) * b;
}


namespace simd
{

//--------------------------------------------------------------------------------------------------
// SIMD intrinsics
//

#define VSNRAY_DEFINE_SELECT__(T)                                               \
MATH_FUNC                                                                       \
inline T select(bool k, T a, T b)                                               \
{                                                                               \
    return k ? a : b;                                                           \
}

VSNRAY_DEFINE_SELECT__(char)
VSNRAY_DEFINE_SELECT__(short)
VSNRAY_DEFINE_SELECT__(int)
VSNRAY_DEFINE_SELECT__(long)
VSNRAY_DEFINE_SELECT__(long long)
VSNRAY_DEFINE_SELECT__(unsigned char)
VSNRAY_DEFINE_SELECT__(unsigned short)
VSNRAY_DEFINE_SELECT__(unsigned int)
VSNRAY_DEFINE_SELECT__(unsigned long)
VSNRAY_DEFINE_SELECT__(unsigned long long)
VSNRAY_DEFINE_SELECT__(float)
VSNRAY_DEFINE_SELECT__(double)
VSNRAY_DEFINE_SELECT__(long double)

#undef VSNRAY_DEFINE_SELECT__

#define VSNRAY_DEFINE_STORE__(T)                                                \
MATH_FUNC                                                                       \
inline void store(T dst[1], T const& src)                                       \
{                                                                               \
    dst[0] = src;                                                               \
}

VSNRAY_DEFINE_STORE__(char)
VSNRAY_DEFINE_STORE__(short)
VSNRAY_DEFINE_STORE__(int)
VSNRAY_DEFINE_STORE__(long)
VSNRAY_DEFINE_STORE__(long long)
VSNRAY_DEFINE_STORE__(unsigned char)
VSNRAY_DEFINE_STORE__(unsigned short)
VSNRAY_DEFINE_STORE__(unsigned int)
VSNRAY_DEFINE_STORE__(unsigned long)
VSNRAY_DEFINE_STORE__(unsigned long long)
VSNRAY_DEFINE_STORE__(float)
VSNRAY_DEFINE_STORE__(double)
VSNRAY_DEFINE_STORE__(long double)

#undef VSNRAY_DEFINE_STORE__

MATH_FUNC
inline bool any(bool b)
{
    return b;
}

MATH_FUNC
inline bool all(bool b)
{
    return b;
}

} // simd



//-------------------------------------------------------------------------------------------------
// Import SIMD intrinsics into namespace visionaray.
// Enable ADL!
//

using simd::select;
using simd::store;
using simd::any;
using simd::all;


//--------------------------------------------------------------------------------------------------
// Masked operations
//

template <typename T, typename M>
MATH_FUNC
inline T neg(T const& a, M const& m)
{
    return select( m, -a, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T add(T const& a, T const& b, M const& m)
{
    return select( m, a + b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T sub(T const& a, T const& b, M const& m)
{
    return select( m, a - b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T mul(T const& a, T const& b, M const& m)
{
    return select( m, a * b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T div(T const& a, T const& b, M const& m)
{
    return select( m, a / b, T(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto add(T1 const& a, T2 const& b, M const& m)
    -> decltype(operator+(a, b))
{
    using T3 = decltype(operator+(a, b));
    return select( m, a + b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto sub(T1 const& a, T2 const& b, M const& m)
    -> decltype(operator-(a, b))
{
    using T3 = decltype(operator-(a, b));
    return select( m, a - b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto mul(T1 const& a, T2 const& b, M const& m)
    -> decltype(operator*(a, b))
{
    using T3 = decltype(operator*(a, b));
    return select( m, a * b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto div(T1 const& a, T2 const& b, M const& m)
    -> decltype(operator/(a, b))
{
    using T3 = decltype(operator/(a, b));
    return select( m, a / b, T3(0.0) );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto add(T1 const& a, T2 const& b, M const& m, T3 const& old = T3(0.0))
    -> decltype(operator+(a, b))
{
    return select( m, a + b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto sub(T1 const& a, T2 const& b, M const& m, T3 const& old = T3(0.0))
    -> decltype(operator-(a, b))
{
    return select( m, a - b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto mul(T1 const& a, T2 const& b, M const& m, T3 const& old = T3(0.0))
    -> decltype(operator*(a, b))
{
    return select( m, a * b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto div(T1 const& a, T2 const& b, M const& m, T3 const& old = T3(0.0))
    -> decltype(operator/(a, b))
{
    return select( m, a / b, old );
}


//--------------------------------------------------------------------------------------------------
// Implement some (useful) functions not defined in <cmath>
//

template <typename T>
MATH_FUNC
inline T fract(T const& x)
{
    return x - floor(x);
}

template <typename T>
MATH_FUNC
inline T heaviside(T const& x)
{
    return select( x < T(0.0), T(0.0), T(1.0) );
}

template <typename T>
MATH_FUNC
inline T clamp(T const& x, T const& a, T const& b)
{
    return max( a, min(x, b) );
}

template <typename T>
MATH_FUNC
inline T saturate(T const& x)
{
    return max(T(0.0), min(x, T(1.0)));
}

template <typename T, typename S>
MATH_FUNC
inline T lerp(T const& a, T const& b, S const& x)
{
    return (S(1.0f) - x) * a + x * b;
}

template <typename T, typename S>
MATH_FUNC
inline T lerp(T const& a, T const& b, T const& c, S const& u, S const& v)
{
    auto s2 = c * v;
    auto s3 = b * u;
    auto s1 = a * (S(1.0f) - (u + v));

    return s1 + s2 + s3;
}

template <typename T>
MATH_FUNC
inline T step(T const& edge, T const& x)
{
    return x < edge ? T(0.0) : T(1.0);
}

template <typename T>
MATH_FUNC
inline T rsqrt(T const& x)
{
    return T(1.0) / sqrt(x);
}

#ifdef __CUDA_ARCH__
MATH_GPU_FUNC
inline float rsqrt(float x)
{
    return rsqrtf(x);
}
#endif

template <typename T>
MATH_FUNC
inline T cot(T const& x)
{
    return T(1.0) / tan(x);
}

template <typename T>
MATH_FUNC
inline T det2(T const& m00, T const& m01, T const& m10, T const& m11)
{
    return m00 * m11 - m10 * m01;
}

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_DETAIL_MATH_H
