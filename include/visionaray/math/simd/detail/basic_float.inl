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
VSNRAY_FORCE_INLINE basic_float<T>& operator+=(basic_float<T>& a, U const& b)
{
    a = a + b;
    return a;
}

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator-=(basic_float<T>& a, U const& b)
{
    a = a - b;
    return a;
}

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator*=(basic_float<T>& a, U const& b)
{
    a = a * b;
    return a;
}

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator/=(basic_float<T>& a, U const& b)
{
    a = a / b;
    return a;
}


//-------------------------------------------------------------------------------------------------
// Bitwise operators
//

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator&=(basic_float<T>& a, U const& b)
{
    a = a & b;
    return a;
}

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator|=(basic_float<T>& a, U const& b)
{
    a = a | b;
    return a;
}

template <typename T, typename U>
VSNRAY_FORCE_INLINE basic_float<T>& operator^=(basic_float<T>& a, U const& b)
{
    a = a ^ b;
    return a;
}


//-------------------------------------------------------------------------------------------------
// Math functions
//

template <typename T>
VSNRAY_FORCE_INLINE basic_float<T> copysign(basic_float<T> const& x, basic_float<T> const& y)
{
    auto xi = reinterpret_as_int(x);
    auto yi = reinterpret_as_int(y);

    xi &= 0x7FFFFFFF;
    xi |= yi & 0x80000000;

    return reinterpret_as_float(xi);
}

} // simd
} // MATH_NAMESPACE
