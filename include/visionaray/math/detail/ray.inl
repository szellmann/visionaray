// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

#include "../simd/type_traits.h"

namespace MATH_NAMESPACE
{

template <typename T>
MATH_FUNC
inline basic_ray<T>::basic_ray(vector<3, T> const& o, vector<3, T> const& d)
    : ori(o)
    , dir(d)
{
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, size_t N>
inline basic_ray<typename float_from_simd_width<N>::type> pack(
        std::array<basic_ray<T>, N> const& rays
        )
{
    using U = typename float_from_simd_width<N>::type;
    using float_array = typename aligned_array<U>::type;

    float_array ori_x;
    float_array ori_y;
    float_array ori_z;

    float_array dir_x;
    float_array dir_y;
    float_array dir_z;

    for (size_t i = 0; i < N; ++i)
    {
        ori_x[i] = rays[i].ori.x;
        ori_y[i] = rays[i].ori.y;
        ori_z[i] = rays[i].ori.z;

        dir_x[i] = rays[i].dir.x;
        dir_y[i] = rays[i].dir.y;
        dir_z[i] = rays[i].dir.z;
    }

    return basic_ray<U>(
            vector<3, U>(ori_x, ori_y, ori_z),
            vector<3, U>(dir_x, dir_y, dir_z)
            );
}

// unpack -------------------------------------------------

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline auto unpack(basic_ray<FloatT> const& ray)
    -> std::array<basic_ray<float>, num_elements<FloatT>::value>
{
    using float_array = typename aligned_array<FloatT>::type;

    float_array ori_x;
    float_array ori_y;
    float_array ori_z;

    float_array dir_x;
    float_array dir_y;
    float_array dir_z;

    store(ori_x, ray.ori.x);
    store(ori_y, ray.ori.y);
    store(ori_z, ray.ori.z);

    store(dir_x, ray.dir.x);
    store(dir_y, ray.dir.y);
    store(dir_z, ray.dir.z);

    std::array<basic_ray<float>, num_elements<FloatT>::value> result;

    for (int i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].ori.x = ori_x[i];
        result[i].ori.y = ori_y[i];
        result[i].ori.z = ori_z[i];

        result[i].dir.x = dir_x[i];
        result[i].dir.y = dir_y[i];
        result[i].dir.z = dir_z[i];
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
