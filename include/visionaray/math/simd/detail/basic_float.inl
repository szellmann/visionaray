// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator+=(basic_float<T>& a, U const& b)
{
    a = a + b;
    return a;
}

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator-=(basic_float<T>& a, U const& b)
{
    a = a - b;
    return a;
}

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator*=(basic_float<T>& a, U const& b)
{
    a = a * b;
    return a;
}

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator/=(basic_float<T>& a, U const& b)
{
    a = a / b;
    return a;
}


//-------------------------------------------------------------------------------------------------
// Bitwise operators
//

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator&=(basic_float<T>& a, U const& b)
{
    a = a & b;
    return a;
}

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator|=(basic_float<T>& a, U const& b)
{
    a = a | b;
    return a;
}

template <typename T, typename U>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T>& operator^=(basic_float<T>& a, U const& b)
{
    a = a ^ b;
    return a;
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

template <typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T> copysign(basic_float<T> const& x, basic_float<T> const& y)
{
    auto xi = reinterpret_as_int(x);
    auto yi = reinterpret_as_int(y);

    xi &= 0x7FFFFFFF;
    xi |= yi & 0x80000000;

    return reinterpret_as_float(xi);
}


//-------------------------------------------------------------------------------------------------
// Newton-Raphson refinement for approximate mathematical functions
//

template <unsigned N, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T> rcp_step(basic_float<T> const& v)
{
    basic_float<T> t = v;

    for (unsigned i = 0; i < N; ++i)
    {
        t = (t + t) - (v * t * t);
    }

    return t;
}

template <unsigned N, typename T>
MATH_FUNC
VSNRAY_FORCE_INLINE basic_float<T> rsqrt_step(basic_float<T> const& v, basic_float<T> const& x0)
{
    basic_float<T> threehalf(1.5f);
    basic_float<T> vhalf = v * basic_float<T>(0.5f);
    basic_float<T> t = x0;

    for (unsigned i = 0; i < N; ++i)
    {
        t = t * (threehalf - vhalf * t * t);
    }

    return t;
}

} // simd
} // MATH_NAMESPACE
