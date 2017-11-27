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

template <typename T>
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

    VSNRAY_FUNC random_sampler() = default;

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

template <>
class random_sampler<simd::float4>
{
public:

    using value_type = simd::float4;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_FUNC random_sampler(array<unsigned, 4> const& seed)
        : samplers_({{ seed[0], seed[1], seed[2], seed[3] }}) // TODO!
    {
    }

    VSNRAY_FUNC simd::float4 next()
    {
        simd::aligned_array_t<value_type> arr;

        for (int i = 0; i < simd::num_elements<simd::float4>::value; ++i)
        {
            arr[i] = samplers_[i].next();
        }

        return simd::float4(arr);
    }

    // TODO: maybe don't have a random_sampler4 at all?
    sampler_type& get_sampler(size_t i)
    {
        return samplers_[i];
    }

private:

    array<sampler_type, simd::num_elements<simd::float4>::value> samplers_;

};

template <>
class random_sampler<simd::float8>
{
public:

    using value_type = simd::float8;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_FUNC random_sampler(array<unsigned, 8> const& seed)
        : samplers_({{ seed[0], seed[1], seed[2], seed[3],
                       seed[4], seed[5], seed[6], seed[7] }}) // TODO!
    {
    }

    VSNRAY_FUNC simd::float8 next()
    {
        simd::aligned_array_t<value_type> arr;

        for (int i = 0; i < simd::num_elements<simd::float8>::value; ++i)
        {
            arr[i] = samplers_[i].next();
        }

        return simd::float8(arr);
    }

    // TODO: maybe don't have a random_sampler8 at all?
    sampler_type& get_sampler(size_t i)
    {
        return samplers_[i];
    }

private:

    array<sampler_type, simd::num_elements<simd::float8>::value> samplers_;

};

template <>
class random_sampler<simd::float16>
{
public:

    using value_type = simd::float16;

public:

    typedef random_sampler<float> sampler_type;

    VSNRAY_FUNC random_sampler(array<unsigned, 16> const& seed)
        : samplers_({{ seed[ 0], seed[ 1], seed[ 2], seed[ 3],
                       seed[ 4], seed[ 5], seed[ 6], seed[ 7],
                       seed[ 8], seed[ 9], seed[10], seed[11],
                       seed[12], seed[13], seed[14], seed[15] }}) // TODO!
    {
    }

    VSNRAY_FUNC simd::float16 next()
    {
        simd::aligned_array_t<value_type> arr;

        for (int i = 0; i < simd::num_elements<simd::float16>::value; ++i)
        {
            arr[i] = samplers_[i].next();
        }

        return simd::float16(arr);
    }

    // TODO: maybe don't have a random_sampler16 at all?
    sampler_type& get_sampler(size_t i)
    {
        return samplers_[i];
    }

private:

    array<sampler_type, simd::num_elements<simd::float16>::value> samplers_;

};

} // visionaray

#endif // VSNRAY_RANDOM_SAMPLER_H
