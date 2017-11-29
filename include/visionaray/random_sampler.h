// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RANDOM_SAMPLER_H
#define VSNRAY_RANDOM_SAMPLER_H 1

#include <chrono>
#include <type_traits>

#if defined(__CUDACC__)
#include <thrust/random.h>
#elif defined(__HCC__)
#include "hcc/random.h"
#else
#include <random>
#endif

#include <visionaray/math/simd/simd.h>
#include <visionaray/array.h>

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// TODO: move to a better place
//

#if defined(__CUDA_ARCH__)
template <
    typename T,
    typename = typename std::enable_if<std::is_floating_point<T>::value>::type
    >
VSNRAY_GPU_FUNC
inline unsigned tic(T /* */)
{
    return clock64();
}
#else
template <
    typename T,
    typename = typename std::enable_if<std::is_floating_point<T>::value>::type
    >
VSNRAY_FUNC
inline unsigned tic(T /* */)
{
    auto t = std::chrono::high_resolution_clock::now();
    return t.time_since_epoch().count();
}
#endif

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_FUNC array<unsigned, simd::num_elements<T>::value> tic(T /* */)
{
    array<unsigned, simd::num_elements<T>::value> result;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        result[i] = tic(float{});
    }

    return result;
}

} // detail


//-------------------------------------------------------------------------------------------------
// random_sampler classes, uses a standard pseudo RNG to generate samples
//

template <typename T, typename = void>
class random_sampler
{
public:

    using value_type = T;

public:

#if defined(__CUDACC__)
    typedef thrust::default_random_engine rand_engine;
    typedef thrust::uniform_real_distribution<T> uniform_dist;
#elif defined(__HCC__)
    typedef hcc::default_random_engine rand_engine;
    typedef hcc::uniform_real_distribution<T> uniform_dist;
#else
    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<T> uniform_dist;
#endif

    random_sampler() = default;

    VSNRAY_FUNC random_sampler(unsigned seed)
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

template <typename T>
class random_sampler<T, typename std::enable_if<simd::is_simd_vector<T>::value>::type>
{
public:

    using value_type = T;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_FUNC random_sampler(array<unsigned, simd::num_elements<value_type>::value> const& seed)
    {
        for (int i = 0; i < simd::num_elements<value_type>::value; ++i)
        {
            samplers_[i] = sampler_type(seed[i]);
        }
    }

    VSNRAY_FUNC value_type next()
    {
        simd::aligned_array_t<value_type> arr;

        for (int i = 0; i < simd::num_elements<value_type>::value; ++i)
        {
            arr[i] = samplers_[i].next();
        }

        return value_type(arr);
    }

    // TODO: maybe don't have a random_samplerN at all?
    sampler_type& get_sampler(size_t i)
    {
        return samplers_[i];
    }

private:

    array<sampler_type, simd::num_elements<value_type>::value> samplers_;

};

} // visionaray

#endif // VSNRAY_RANDOM_SAMPLER_H
