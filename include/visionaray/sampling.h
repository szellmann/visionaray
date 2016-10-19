// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SAMPLING_H
#define VSNRAY_SAMPLING_H 1

#include <visionaray/math/math.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Compute the radical inverse of a nonnegative integer
//
// d1|d2|d3|d4 ==> 0.d4|d3|d2|d1, where di in [0..Base)
//

template <
    unsigned Base,
    typename I,
    typename F = simd::float_type_t<I>
    >
VSNRAY_FUNC
inline F radical_inverse(I n)
{
    F result(0.0);
    F inv_base(1.0f / Base);
    F inv_bi = inv_base;

    while (any(n > 0))
    {
        F digit = convert_to_float(n % Base);
        result += digit * inv_bi;
        n /= I(Base);
        inv_bi *= inv_base;
    }

    return result;
}

} // detail


//-------------------------------------------------------------------------------------------------
// Utility functions for geometry sampling
//

template <typename T>
VSNRAY_FUNC
inline vector<3, T> uniform_sample_hemisphere(T u1, T u2)
{
    auto r   = sqrt( max(T(0.0), T(1.0) - u1 * u1) );
    auto phi = constants::two_pi<T>() * u2;
    return vector<3, T>(r * cos(phi), r * sin(phi), u1);
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> cosine_sample_hemisphere(T u1, T u2)
{
    auto r     = sqrt(u1);
    auto theta = constants::two_pi<T>() * u2;
    auto x     = r * cos(theta);
    auto y     = r * sin(theta);
    auto z     = sqrt( max(T(0.0), T(1.0) - u1) );
    return vector<3, T>(x, y, z);
}

} // visionaray

#endif // VSNRAY_SAMPLING_H
