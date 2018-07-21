// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SAMPLING_H
#define VSNRAY_SAMPLING_H 1

#include <type_traits>

#include "math/detail/math.h"
#include "math/simd/type_traits.h"
#include "math/array.h"
#include "math/constants.h"
#include "math/vector.h"
#include "light_sample.h"

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
inline vector<2, T> concentric_sample_disk(T const& u1, T const& u2)
{
    // http://psgraphics.blogspot.de/2011/01/improved-code-for-concentric-map.html

    T a = T(2.0) * u1 - T(1.0);
    T b = T(2.0) * u2 - T(1.0);

//  auto cond = abs(a) > abs(b);
    auto cond = a * a > b * b;

    T r   = select(cond, a, b);
    T phi = select(
            cond,
            constants::pi_over_four<T>() * (b / a),
            constants::pi_over_two<T>() - constants::pi_over_four<T>() * (a / b)
            );

    return vector<2, T>(r * cos(phi), r * sin(phi));
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> uniform_sample_hemisphere(T const& u1, T const& u2)
{
    auto r   = sqrt( max(T(0.0), T(1.0) - u1 * u1) );
    auto phi = constants::two_pi<T>() * u2;
    return vector<3, T>(r * cos(phi), r * sin(phi), u1);
}

template <typename T>
VSNRAY_FUNC
inline vector<3, T> cosine_sample_hemisphere(T const& u1, T const& u2)
{
    auto r     = sqrt(u1);
    auto theta = constants::two_pi<T>() * u2;
    auto x     = r * cos(theta);
    auto y     = r * sin(theta);
    auto z     = sqrt( max(T(0.0), T(1.0) - u1) );
    return vector<3, T>(x, y, z);
}


//-------------------------------------------------------------------------------------------------
// Sample a random light
//

// non-simd
template <
    typename Lights,
    typename Generator,
    typename T = typename Generator::value_type,
    typename = typename std::enable_if<!simd::is_simd_vector<T>::value>::type
    >
VSNRAY_FUNC
light_sample<T> sample_random_light(Lights begin, Lights end, Generator& gen)
{
    auto num_lights = end - begin;

    auto u = gen.next();

    int light_id = static_cast<int>(u * num_lights);

    return begin[light_id].sample(gen);
}

// simd
template <
    typename Lights,
    typename Generator,
    typename T = typename Generator::value_type,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
light_sample<T> sample_random_light(Lights begin, Lights end, Generator& gen, T = T())
{
    using float_array = simd::aligned_array_t<T>;

    auto num_lights = end - begin;

    auto u = gen.next();

    float_array uf;
    store(uf, u);

    array<vector<3, float>, simd::num_elements<T>::value> poss;
    array<vector<3, float>, simd::num_elements<T>::value> intensities;
    array<vector<3, float>, simd::num_elements<T>::value> normals;
    array<float, simd::num_elements<T>::value> areas;

    for (size_t i = 0; i < simd::num_elements<T>::value; ++i)
    {
        int light_id = static_cast<int>(uf[i] * num_lights);

        auto ls = begin[light_id].sample(gen.get_generator(i));

        poss[i] = ls.pos;
        intensities[i] = ls.intensity;
        normals[i] = ls.normal;
        areas[i] = ls.area;
    }

    light_sample<T> result;

    result.pos = simd::pack(poss);
    result.intensity = simd::pack(intensities);
    result.normal = simd::pack(normals);
    result.area = T(areas.data());

    return result;
}

} // visionaray

#endif // VSNRAY_SAMPLING_H
