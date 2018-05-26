// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RANDOM_GENERATOR_H
#define VSNRAY_RANDOM_GENERATOR_H 1

#include <chrono>
#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
#include <thrust/random.h>
#elif defined(__HCC__)
#include "hcc/random.h"
#else
#include <random>
#endif

#include "math/simd/type_traits.h"
#include "math/array.h"

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
#elif defined(__KALMAR_ACCELERATOR__)
template <
    typename T,
    typename = typename std::enable_if<std::is_floating_point<T>::value>::type
    >
VSNRAY_FUNC
inline unsigned tic(T /* */)
{
    return {}; // TODO!
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
// random_generator classes, uses a standard pseudo RNG to generate samples
//

template <typename T, typename = void>
class random_generator
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

    random_generator() = default;

    VSNRAY_FUNC random_generator(unsigned seed)
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
class random_generator<T, typename std::enable_if<simd::is_simd_vector<T>::value>::type>
{
public:

    using value_type = T;

public:

    typedef random_generator<float> generator_type;

    VSNRAY_FUNC random_generator(array<unsigned, simd::num_elements<value_type>::value> const& seed)
    {
        for (int i = 0; i < simd::num_elements<value_type>::value; ++i)
        {
            generators_[i] = generator_type(seed[i]);
        }
    }

    VSNRAY_FUNC value_type next()
    {
        simd::aligned_array_t<value_type> arr;

        for (int i = 0; i < simd::num_elements<value_type>::value; ++i)
        {
            arr[i] = generators_[i].next();
        }

        return value_type(arr);
    }

    // TODO: maybe don't have a random_generatorN at all?
    VSNRAY_FUNC generator_type& get_generator(size_t i)
    {
        return generators_[i];
    }

private:

    array<generator_type, simd::num_elements<value_type>::value> generators_;

};

} // visionaray

#endif // VSNRAY_RANDOM_GENERATOR_H
