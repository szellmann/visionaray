// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Bitwise operators
//

template <typename ...Args, typename T>
VSNRAY_FORCE_INLINE basic_mask<Args...>& operator&=(basic_mask<Args...>& a, T const& b)
{
    a = a & b;
    return a;
}

template <typename ...Args, typename T>
VSNRAY_FORCE_INLINE basic_mask<Args...>& operator|=(basic_mask<Args...>& a, T const& b)
{
    a = a | b;
    return a;
}

template <typename ...Args, typename T>
VSNRAY_FORCE_INLINE basic_mask<Args...>& operator^=(basic_mask<Args...>& a, T const& b)
{
    a = a ^ b;
    return a;
}

} // simd
} // MATH_NAMESPACE
