// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_SIMD_RANDOM_H
#define VSNRAY_MATH_SIMD_RANDOM_H 1

#include "../config.h"
#include "type_traits.h"

#include <type_traits>

namespace MATH_NAMESPACE
{
namespace simd
{

template <
    typename T,
    typename UI,
    UI a,
    UI c,
    UI m,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<std::is_integral<UI>::value>::type
    >
class linear_congruential_engine
{
public:

    using result_type = T;

    const static UI multiplier   = a;
    const static UI increment    = c;
    const static UI modulus      = m;
    const static UI default_seed = 1u;

public:

    MATH_FUNC
    linear_congruential_engine(T const& seed = T(default_seed))
        : x(seed)
    {
    }

    MATH_FUNC
    T operator()()
    {
        x = (a * x + c) % m;
        return x;
    }

private:

    T x;

};

template <
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type
    >
using minstd_rand0 = linear_congruential_engine<T, unsigned, 16807u, 0u, 2147483647u>;

} // simd
} // MATH_NAMESPACE

#endif // VSNRAY_MATH_SIMD_RANDOM_H
