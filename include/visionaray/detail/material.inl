// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>

#include "../array.h"

namespace visionaray
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack N materials into a single SIMD material
//

template <size_t N>
VSNRAY_FUNC
inline emissive<float_from_simd_width_t<N>> pack(array<emissive<float>, N> const& mats)
{
    using T = float_from_simd_width_t<N>;

    emissive<T> result;

    float* ls = reinterpret_cast<float*>(&result.ls());

    for (size_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float* ce_j = reinterpret_cast<float*>(&result.ce()[j]);
            ce_j[i] = mats[i].ce()[j];
        }
        ls[i] = mats[i].ls();
    }

    return result;
}

template <size_t N>
VSNRAY_FUNC
inline matte<float_from_simd_width_t<N>> pack(array<matte<float>, N> const& mats)
{
    using T = float_from_simd_width_t<N>;

    matte<T> result;

    float* ka = reinterpret_cast<float*>(&result.ka());
    float* kd = reinterpret_cast<float*>(&result.kd());

    for (size_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float* ca_j = reinterpret_cast<float*>(&result.ca()[j]);
            float* cd_j = reinterpret_cast<float*>(&result.cd()[j]);
            ca_j[i] = mats[i].ca()[j];
            cd_j[i] = mats[i].cd()[j];
        }
        ka[i] = mats[i].ka();
        kd[i] = mats[i].kd();
    }

    return result;
}

template <size_t N>
VSNRAY_FUNC
inline mirror<float_from_simd_width_t<N>> pack(array<mirror<float>, N> const& mats)
{
    using T = float_from_simd_width_t<N>;

    mirror<T> result;

    float* kr = reinterpret_cast<float*>(&result.kr());

    for (size_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float* cr_j = reinterpret_cast<float*>(&result.cr()[j]);
            cr_j[i] = mats[i].cr()[j];
        }
        kr[i] = mats[i].kr();
    }

    return result;
}

template <size_t N>
VSNRAY_FUNC
inline plastic<float_from_simd_width_t<N>> pack(array<plastic<float>, N> const& mats)
{
    using T = float_from_simd_width_t<N>;

    plastic<T> result;

    float* ka = reinterpret_cast<float*>(&result.ka());
    float* kd = reinterpret_cast<float*>(&result.kd());
    float* ks = reinterpret_cast<float*>(&result.ks());
    float* se = reinterpret_cast<float*>(&result.specular_exp());

    for (size_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float* ca_j = reinterpret_cast<float*>(&result.ca()[j]);
            float* cd_j = reinterpret_cast<float*>(&result.cd()[j]);
            float* cs_j = reinterpret_cast<float*>(&result.cs()[j]);
            ca_j[i] = mats[i].ca()[j];
            cd_j[i] = mats[i].cd()[j];
            cs_j[i] = mats[i].cs()[j];
        }
        ka[i] = mats[i].ka();
        kd[i] = mats[i].kd();
        ks[i] = mats[i].ks();
        se[i] = mats[i].specular_exp();
    }

    return result;
}

} // simd
} // visioanray
