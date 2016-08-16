// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MC_H
#define VSNRAY_MC_H 1

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
class sampler
{
public:

    using value_type = T;

public:

#ifdef __CUDA_ARCH__
    typedef thrust::default_random_engine rand_engine;
    typedef thrust::uniform_real_distribution<T> uniform_dist;
#else
    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<T> uniform_dist;
#endif

    VSNRAY_FUNC sampler() = default;

    VSNRAY_FUNC sampler(unsigned seed)
        : rng_(rand_engine(seed))
        , dist_(uniform_dist(0, 1))
    {
    }

    VSNRAY_FUNC T next()
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

    using value_type = simd::float4;

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

    // TODO: maybe don't have a sampler4 at all?
    sampler_type& get_sampler()
    {
        return sampler_;
    }

private:

    sampler_type sampler_;
};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
template <>
class sampler<simd::float8>
{
public:

    using value_type = simd::float8;

public:

    typedef sampler<float> sampler_type;

    VSNRAY_CPU_FUNC sampler(unsigned seed)
        : sampler_(seed)
    {
    }

    VSNRAY_CPU_FUNC simd::float8 next()
    {
        return simd::float8
        (
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next(),
            sampler_.next()
        );
    }

    // TODO: maybe don't have a sampler4 at all?
    sampler_type& get_sampler()
    {
        return sampler_;
    }

private:

    sampler_type sampler_;
};
#endif


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

#endif // VSNRAY_MC_H
