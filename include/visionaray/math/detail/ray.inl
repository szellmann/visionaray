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

} // simd
} // MATH_NAMESPACE
