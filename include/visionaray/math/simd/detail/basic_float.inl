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

} // simd
} // MATH_NAMESPACE
