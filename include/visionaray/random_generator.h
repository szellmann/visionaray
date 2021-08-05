// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RANDOM_GENERATOR_H
#define VSNRAY_RANDOM_GENERATOR_H 1

#include <visionaray/config.h>

#include <type_traits>

#if defined(__CUDACC__) && VSNRAY_HAVE_THRUST
#include <thrust/random.h>
#else
#include <random>
#endif

#include "detail/macros.h"
#include "math/simd/type_traits.h"
#include "array.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// random_generator classes, uses a standard pseudo RNG to generate samples
//

template <typename T, typename = void>
class random_generator
{
public:

    using value_type = T;

public:

#if defined(__CUDACC__) && VSNRAY_HAVE_THRUST
    typedef thrust::default_random_engine rand_engine;
    typedef thrust::uniform_real_distribution<T> uniform_dist;
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
    VSNRAY_FUNC generator_type& get_generator(unsigned i)
    {
        return generators_[i];
    }

private:

    array<generator_type, simd::num_elements<value_type>::value> generators_;

};

} // visionaray

#endif // VSNRAY_RANDOM_GENERATOR_H
