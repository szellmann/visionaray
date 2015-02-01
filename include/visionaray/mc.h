// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_MC_H
#define VSNRAY_DETAIL_MC_H

#ifdef __CUDA_ARCH__
#include <thrust/random.h>
#else
#include <random>
#endif

#include <visionaray/math/math.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// sampler classes
//

template <typename T>
class sampler;

template <>
class sampler<float>
{
public:

#ifdef __CUDA_ARCH__
    typedef thrust::default_random_engine rand_engine;
    typedef thrust::uniform_real_distribution<float> uniform_dist;
#else
    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<float> uniform_dist;
#endif

    VSNRAY_FUNC sampler(unsigned seed)
        : rng_(rand_engine(seed))
        , dist_(uniform_dist(0, 1))
    {
    }

    VSNRAY_FUNC float next()
    {
        return dist_(rng_);
    }

private:

    rand_engine  rng_;
    uniform_dist dist_;

};

template <>
class sampler<simd::float4>
{
public:

    typedef sampler<float> sampler_type;

    VSNRAY_CPU_FUNC sampler(unsigned seed)
        : sampler_(seed)
    {
    }

    VSNRAY_CPU_FUNC simd::float4 next()
    {
        return simd::float4
        (
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next()
        );
    }

private:

    sampler_type sampler_;
};


//-------------------------------------------------------------------------------------------------
// Utility functions for Geometry sampling
//

template <template <typename> class S, typename T>
VSNRAY_FUNC
vector<3, T> sample_hemisphere(S<T>& sampler)
{
    auto sample = vector<2, T>(sampler.next(), sampler.next());
    auto cosphi = cos(T(2.0) * constants::pi<T>() * sample.x);
    auto sinphi = sin(T(2.0) * constants::pi<T>() * sample.x);
    auto costheta = T(1.0) - sample.y;
    auto sintheta = sqrt(T(1.0) - costheta * costheta);
    return vector<3, T>(sintheta * cosphi, sintheta * sinphi, costheta);
}

template <typename S, typename T>
VSNRAY_FUNC
vector<3, T> sample_hemisphere(T e, S& sampler)
{
    auto sample = vector<2, T>(sampler.next(), sampler.next());
    auto cosphi = cos(T(2.0) * constants::pi<T>() * sample.x);
    auto sinphi = sin(T(2.0) * constants::pi<T>() * sample.x);
    auto costheta = pow((T(1.0) - sample.y), T(1.0) / (e + T(1.0)));
    auto sintheta = sqrt(T(1.0) - costheta * costheta);
    return vector<3, T>(sintheta * cosphi, sintheta * sinphi, costheta);
}

} // visionaray

#endif // VSNRAY_DETAIL_MC_H


