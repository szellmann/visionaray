// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RANDOM_SAMPLER_H
#define VSNRAY_RANDOM_SAMPLER_H 1

#if defined(__CUDA_ARCH__)
#include <thrust/random.h>
#elif defined(__KALMAR_ACCELERATOR__)
#include "hcc/random.h"
#else
#include <random>
#endif

#include <visionaray/math/simd/simd.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// random_sampler classes, uses a standard pseudo RNG to generate samples
//

template <typename T>
class random_sampler
{
public:

    using value_type = T;

public:

#if defined(__CUDA_ARCH__)
    typedef thrust::default_random_engine rand_engine;
    typedef thrust::uniform_real_distribution<T> uniform_dist;
#elif defined(__KALMAR_ACCELERATOR__)
    typedef hcc::default_random_engine rand_engine;
    typedef hcc::uniform_real_distribution<T> uniform_dist;
#else
    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<T> uniform_dist;
#endif

// TODO: avoid code duplication here (somehow)
#if VSNRAY_GPU_MODE
    VSNRAY_GPU_FUNC random_sampler() = default;

    VSNRAY_GPU_FUNC random_sampler(unsigned seed)
        : rng_(rand_engine(seed))
        , dist_(uniform_dist(0, 1))
    {
    }

    VSNRAY_GPU_FUNC T next()
    {
        return dist_(rng_);
    }
#else
    random_sampler() = default;

    random_sampler(unsigned seed)
        : rng_(rand_engine(seed))
        , dist_(uniform_dist(0, 1))
    {
    }

    T next()
    {
        return dist_(rng_);
    }
#endif

private:

    rand_engine  rng_;
    uniform_dist dist_;

};

template <>
class random_sampler<simd::float4>
{
public:

    using value_type = simd::float4;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_FUNC random_sampler(unsigned seed)
        : sampler_(seed)
    {
    }

    VSNRAY_FUNC simd::float4 next()
    {
        return simd::float4(
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next()
                );
    }

    // TODO: maybe don't have a random_sampler4 at all?
    sampler_type& get_sampler()
    {
        return sampler_;
    }

private:

    sampler_type sampler_;
};

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
template <>
class random_sampler<simd::float8>
{
public:

    using value_type = simd::float8;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_CPU_FUNC random_sampler(unsigned seed)
        : sampler_(seed)
    {
    }

    VSNRAY_CPU_FUNC simd::float8 next()
    {
        return simd::float8(
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

    // TODO: maybe don't have a random_sampler8 at all?
    sampler_type& get_sampler()
    {
        return sampler_;
    }

private:

    sampler_type sampler_;
};
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
template <>
class random_sampler<simd::float16>
{
public:

    using value_type = simd::float16;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_CPU_FUNC random_sampler(unsigned seed)
        : sampler_(seed)
    {
    }

    VSNRAY_CPU_FUNC simd::float16 next()
    {
        return simd::float16(
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
                sampler_.next(),
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

    // TODO: maybe don't have a random_sampler16 at all?
    sampler_type& get_sampler()
    {
        return sampler_;
    }

private:

    sampler_type sampler_;
};
#endif

} // visionaray

#endif // VSNRAY_RANDOM_SAMPLER_H
