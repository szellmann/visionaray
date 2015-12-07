// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_NORM_H
#define VSNRAY_MATH_NORM_H 1

#include <cstdint>

namespace MATH_NAMESPACE
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Best integer type from bit depth. TODO: support arbitrary bit depth
//

template <unsigned Bits> struct best_uint { /* No best_int::type causes a compiler error */ };
template <> struct best_uint< 8> { typedef uint8_t  type; };
template <> struct best_uint<16> { typedef uint16_t type; };
template <> struct best_uint<32> { typedef uint32_t type; };


//-------------------------------------------------------------------------------------------------
// Convert float to unorm (cf. OpenGL 4.4, 2.3.4.1
//

template <unsigned Bits>
MATH_FUNC
inline uint32_t float_to_unorm(float f)
{
    f = saturate(f);
    return static_cast<uint32_t>(f * ((1 << Bits) - 1));
}

template <unsigned Bits>
MATH_FUNC
inline float unorm_to_float(uint32_t u)
{
    return static_cast<float>(u) / ((1 << Bits) - 1);
}

} // detail

template <unsigned Bits>
class unorm
{
public:

    using value_type = typename detail::best_uint<Bits>::type;

    MATH_FUNC unorm() = default;

    MATH_FUNC
    /* implicit */ unorm(float f)
        : value(detail::float_to_unorm<Bits>(f))
    {
    }

    MATH_FUNC
    operator value_type() const
    {
        return value;
    }

    MATH_FUNC
    operator float() const
    {
        return detail::unorm_to_float<Bits>(value);
    }

    value_type value;

};

} // MATH_NAMESPACE

#endif // VSNRAY_MATH_NORM_H
