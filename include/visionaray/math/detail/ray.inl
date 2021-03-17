// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include "../simd/type_traits.h"
#include "../config.h"
#include "../limits.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Constructors
//

template <typename T>
MATH_FUNC
inline basic_ray<T>::basic_ray(vector<3, T> const& o, vector<3, T> const& d)
    : ori(o)
    , dir(d)
    , tmin(T(0.0))
    , tmax(numeric_limits<T>::max())
{
}

template <typename T>
MATH_FUNC
inline basic_ray<T>::basic_ray(
        vector<3, T> const& o,
        vector<3, T> const& d,
        T const& tmin,
        T const& tmax
        )
    : ori(o)
    , dir(d)
    , tmin(tmin)
    , tmax(tmax)
{
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, size_t N>
inline auto pack(array<basic_ray<T>, N> const& rays)
    -> basic_ray<float_from_simd_width_t<N>>
{
    using U = float_from_simd_width_t<N>;
    using float_array = aligned_array_t<U>;

    float_array ori_x;
    float_array ori_y;
    float_array ori_z;

    float_array dir_x;
    float_array dir_y;
    float_array dir_z;

    float_array tmin;
    float_array tmax;

    for (size_t i = 0; i < N; ++i)
    {
        ori_x[i] = rays[i].ori.x;
        ori_y[i] = rays[i].ori.y;
        ori_z[i] = rays[i].ori.z;

        dir_x[i] = rays[i].dir.x;
        dir_y[i] = rays[i].dir.y;
        dir_z[i] = rays[i].dir.z;

        tmin[i] = rays[i].tmin;
        tmax[i] = rays[i].tmax;
    }

    return basic_ray<U>(
            vector<3, U>(ori_x, ori_y, ori_z),
            vector<3, U>(dir_x, dir_y, dir_z),
            tmin,
            tmax
            );
}

// pack four rays

template <typename T>
inline auto pack(
        basic_ray<T> const& r1,
        basic_ray<T> const& r2,
        basic_ray<T> const& r3,
        basic_ray<T> const& r4
        )
    -> basic_ray<float_from_simd_width_t<4>>
{
    return pack( array<basic_ray<T>, 4>{{
            r1, r2, r3, r4
            }} );
}

// pack eight rays

template <typename T>
inline auto pack(
        basic_ray<T> const& r1,
        basic_ray<T> const& r2,
        basic_ray<T> const& r3,
        basic_ray<T> const& r4,
        basic_ray<T> const& r5,
        basic_ray<T> const& r6,
        basic_ray<T> const& r7,
        basic_ray<T> const& r8
        )
    -> basic_ray<float_from_simd_width_t<8>>
{
    return pack( array<basic_ray<T>, 8>{{
            r1, r2, r3, r4, r5, r6, r7, r8
            }} );
}

// unpack -------------------------------------------------

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline auto unpack(basic_ray<FloatT> const& ray)
    -> array<basic_ray<float>, num_elements<FloatT>::value>
{
    using float_array = aligned_array_t<FloatT>;

    float_array ori_x;
    float_array ori_y;
    float_array ori_z;

    float_array dir_x;
    float_array dir_y;
    float_array dir_z;

    float_array tmin;
    float_array tmax;

    store(ori_x, ray.ori.x);
    store(ori_y, ray.ori.y);
    store(ori_z, ray.ori.z);

    store(dir_x, ray.dir.x);
    store(dir_y, ray.dir.y);
    store(dir_z, ray.dir.z);

    store(tmin, ray.tmin);
    store(tmax, ray.tmax);

    array<basic_ray<float>, num_elements<FloatT>::value> result;

    for (int i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].ori.x = ori_x[i];
        result[i].ori.y = ori_y[i];
        result[i].ori.z = ori_z[i];

        result[i].dir.x = dir_x[i];
        result[i].dir.y = dir_y[i];
        result[i].dir.z = dir_z[i];

        result[i].tmin = tmin[i];
        result[i].tmax = tmax[i];
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
