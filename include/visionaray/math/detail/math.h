// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_DETAIL_MATH_H
#define VSNRAY_DETAIL_MATH_H

#include <cmath>
#include <type_traits>

#include "../config.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// Import required math functions from the standard library.
// Enable ADL!
//

using std::abs;
using std::cos;
using std::pow;
using std::sin;
using std::sqrt;

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

#define FF(x) __int_as_float(x)
#define II(x) __float_as_int(x)

MATH_GPU_FUNC
inline float min(float x, float y, float z)
{
    return FF( min(II(x), II(y), II(z)) );
}

MATH_GPU_FUNC
inline float max(float x, float y, float z)
{
    return FF( max(II(x), II(y), II(z)) );
}

MATH_GPU_FUNC
inline float min_max(float x, float y, float z)
{
    return FF( min_max(II(x), II(y), II(z)) );
}

MATH_GPU_FUNC
inline float max_min(float x, float y, float z)
{
    return FF( max_min(II(x), II(y), II(z)) );
}

#undef II
#undef FF

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


//--------------------------------------------------------------------------------------------------
// Constants
//

namespace constants
{
template <typename T> MATH_FUNC T degrees_to_radians()  { return T(1.74532925199432957692369076849e-02); }
template <typename T> MATH_FUNC T radians_to_degrees()  { return T(5.72957795130823208767981548141e+01); }
template <typename T> MATH_FUNC T e()                   { return T(2.71828182845904523536028747135e+00); }
template <typename T> MATH_FUNC T log2_e()              { return T(1.44269504088896338700465094007e+00); }
template <typename T> MATH_FUNC T pi()                  { return T(3.14159265358979323846264338328e+00); }
template <typename T> MATH_FUNC T two_pi()              { return T(6.28318530717958647692528676656e+00); }
template <typename T> MATH_FUNC T inv_pi()              { return T(3.18309886183790691216444201928e-01); }
} // constants


//--------------------------------------------------------------------------------------------------
// Masked operations
//

template <typename T, typename M>
MATH_FUNC
inline T select(M k, T const& a, T const& b)
{
    return k ? a : b;
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline bool select(M k, T1 const& a, T2 const& b)
{
    return k ? a : b;
}

template <typename T, typename M>
MATH_FUNC
inline T neg(T a, M m)
{
    return select( m, -a, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T add(T a, T b, M m)
{
    return select( m, a + b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T sub(T a, T b, M m)
{
    return select( m, a - b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T mul(T a, T b, M m)
{
    return select( m, a * b, T(0.0) );
}

template <typename T, typename M>
MATH_FUNC
inline T div(T a, T b, M m)
{
    return select( m, a / b, T(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto add(T1 a, T2 b, M m) -> decltype(operator+(a, b))
{
    using T3 = decltype(operator+(a, b));
    return select( m, a + b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto sub(T1 a, T2 b, M m) -> decltype(operator-(a, b))
{
    using T3 = decltype(operator-(a, b));
    return select( m, a - b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto mul(T1 a, T2 b, M m) -> decltype(operator*(a, b))
{
    using T3 = decltype(operator*(a, b));
    return select( m, a * b, T3(0.0) );
}

template <typename T1, typename T2, typename M>
MATH_FUNC
inline auto div(T1 a, T2 b, M m) -> decltype(operator/(a, b))
{
    using T3 = decltype(operator/(a, b));
    return select( m, a / b, T3(0.0) );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto add(T1 a, T2 b, M m, T3 old = T3(0.0)) -> decltype(operator+(a, b))
{
    return select( m, a + b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto sub(T1 a, T2 b, M m, T3 old = T3(0.0)) -> decltype(operator-(a, b))
{
    return select( m, a - b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto mul(T1 a, T2 b, M m, T3 old = T3(0.0)) -> decltype(operator*(a, b))
{
    return select( m, a * b, old );
}

template <typename T1, typename T2, typename T3, typename M>
MATH_FUNC
inline auto div(T1 a, T2 b, M m, T3 old = T3(0.0)) -> decltype(operator/(a, b))
{
    return select( m, a / b, old );
}

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


//--------------------------------------------------------------------------------------------------
// Implement some (useful) functions not defined in <cmath>
//

template <typename T>
MATH_FUNC
inline T heavyside(T x)
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

template <typename T>
MATH_FUNC
inline T rsqrt(T const& x)
{
    return T(1.0) / sqrt(x);
}

template <typename T>
MATH_FUNC
inline T cot(T x)
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


#endif // VSNRAY_DETAIL_MATH_H
