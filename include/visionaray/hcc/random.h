// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_RANDOM_H
#define VSNRAY_HCC_RANDOM_H 1

#include <type_traits>

#include "../detail/macros.h"
#include "utility.h"

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// linear_congruential_engine interface
//

template <typename UI, UI a, UI c, UI m>
class linear_congruential_engine
{
public:

    static_assert(std::is_integral<UI>::value, "Type must be integral");
    static_assert(std::is_unsigned<UI>::value, "Type must be unsigned");

    typedef UI result_type;

    static const UI multiplier = a;
    static const UI increment = c;
    static const UI modulus = m;

    static const UI min = c == 0U ? 1U : 0U;
    static const UI max = m - 1U;

    static const UI default_seed = 1U;

public:

    VSNRAY_FUNC
    explicit linear_congruential_engine(UI s = default_seed);

    VSNRAY_FUNC
    void seed(UI s = default_seed);

    VSNRAY_FUNC
    result_type operator()();

private:

    UI x_;

};


//-------------------------------------------------------------------------------------------------
// uniform_real_distribution interface
//

template <typename RealType = double>
class uniform_real_distribution
{
public:

    typedef RealType result_type;
    typedef hcc::pair<RealType, RealType> param_type;

public:

    VSNRAY_FUNC
    explicit uniform_real_distribution(RealType a = 0.0, RealType b = 0.0);

    VSNRAY_FUNC
    explicit uniform_real_distribution(param_type const& param);

    template <typename UniformRNG>
    VSNRAY_FUNC
    result_type operator()(UniformRNG& urng);

    template <typename UniformRNG>
    VSNRAY_FUNC
    result_type operator()(UniformRNG& urng, param_type const& param);

    VSNRAY_FUNC
    result_type a() const;

    VSNRAY_FUNC
    result_type b() const;

    VSNRAY_FUNC
    param_type param() const;

    VSNRAY_FUNC
    void param(param_type const& param);

    VSNRAY_FUNC
    result_type min() const;

    VSNRAY_FUNC
    result_type max() const;

private:

    param_type param_;

};


//-------------------------------------------------------------------------------------------------
// typedefs
//

using minstd_rand0 = linear_congruential_engine<unsigned, 16807u, 0u, 2147483647u>;

using default_random_engine = minstd_rand0;

} // hcc
} // visionaray

#include "detail/linear_congruential_engine.inl"
#include "detail/uniform_real_distribution.inl"

#endif // VSNRAY_HCC_RANDOM_H
