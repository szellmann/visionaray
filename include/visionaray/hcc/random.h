// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_HCC_RANDOM_H
#define VSNRAY_HCC_RANDOM_H 1

#include "../detail/macros.h"
#include "utility.h"

namespace visionaray
{
namespace hcc
{

template <typename RealType = double>
class uniform_real_distribution
{
public:

    typedef RealType result_type;
    typedef hcc::pair<RealType, RealType> param_type;

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

} // hcc
} // visionaray

#include "detail/uniform_real_distribution.inl"

#endif // VSNRAY_HCC_RANDOM_H
