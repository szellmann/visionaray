// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_NORM_H
#define VSNRAY_NORM_H

namespace visionaray
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
VSNRAY_FUNC
inline uint32_t float_to_unorm(float f)
{
    return static_cast<uint32_t>(f * ((2 << (Bits - 1)) - 1));
}

template <unsigned Bits>
VSNRAY_FUNC
inline float unorm_to_float(uint32_t u)
{
    return static_cast<float>(u) / ((2 << (Bits - 1)) - 1);
}

} // detail

template <unsigned Bits>
class unorm
{
public:

    using value_type = typename detail::best_uint<Bits>::type;

    VSNRAY_FUNC unorm() = default;

    VSNRAY_FUNC
    /* implicit */ unorm(float f)
        : value(detail::float_to_unorm<Bits>(f))
    {
    }

    VSNRAY_FUNC
    operator float() const
    {
        return detail::unorm_to_float<Bits>(value);
    }

    value_type value;

};

} // visionaray

#endif // VSNRAY_NORM_H
