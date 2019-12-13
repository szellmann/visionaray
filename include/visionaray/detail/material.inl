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

// emissive -----------------------------------------------

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

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
VSNRAY_FUNC
inline auto unpack(emissive<FloatT> const& mat)
    -> array<emissive<float>, num_elements<FloatT>::value>
{
    array<emissive<float>, num_elements<FloatT>::value> result;

    float const* ls = reinterpret_cast<float const*>(&mat.ls());

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float const* ce_j = reinterpret_cast<float const*>(&mat.ce()[j]);
            result[i].ce()[j] = ce_j[i];
        }
        result[i].ls() = ls[i];
    }

    return result;
}

// matte --------------------------------------------------

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

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
VSNRAY_FUNC
inline auto unpack(matte<FloatT> const& mat)
    -> array<matte<float>, num_elements<FloatT>::value>
{
    array<matte<float>, num_elements<FloatT>::value> result;

    float const* ka = reinterpret_cast<float const*>(&mat.ka());
    float const* kd = reinterpret_cast<float const*>(&mat.kd());

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float const* ca_j = reinterpret_cast<float const*>(&mat.ca()[j]);
            float const* cd_j = reinterpret_cast<float const*>(&mat.cd()[j]);
            result[i].ca()[j] = ca_j[i];
            result[i].cd()[j] = cd_j[i];
        }
        result[i].ka() = ka[i];
        result[i].kd() = kd[i];
    }

    return result;
}

// mirror -------------------------------------------------

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
            float* ior_j = reinterpret_cast<float*>(&result.ior()[j]);
            float* abs_j = reinterpret_cast<float*>(&result.absorption()[j]);
            cr_j[i] = mats[i].cr()[j];
            ior_j[i] = mats[i].ior()[j];
            abs_j[i] = mats[i].absorption()[j];
        }
        kr[i] = mats[i].kr();
    }

    return result;
}

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
VSNRAY_FUNC
inline auto unpack(mirror<FloatT> const& mat)
    -> array<mirror<float>, num_elements<FloatT>::value>
{
    array<mirror<float>, num_elements<FloatT>::value> result;

    float const* kr = reinterpret_cast<float const*>(&mat.kr());

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float const* cr_j = reinterpret_cast<float const*>(&mat.cr()[j]);
            float const* ior_j = reinterpret_cast<float const*>(&mat.ior()[j]);
            float const* abs_j = reinterpret_cast<float const*>(&mat.absorption()[j]);
            result[i].cr()[j] = cr_j[i];
            result[i].ior()[j] = ior_j[i];
            result[i].absorption()[j] = abs_j[i];
        }
        result[i].kr() = kr[i];
    }

    return result;
}

// plastic ------------------------------------------------

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

template <
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
VSNRAY_FUNC
inline auto unpack(plastic<FloatT> const& mat)
    -> array<plastic<float>, num_elements<FloatT>::value>
{
    array<plastic<float>, num_elements<FloatT>::value> result;

    float const* ka = reinterpret_cast<float const*>(&mat.ka());
    float const* kd = reinterpret_cast<float const*>(&mat.kd());
    float const* ks = reinterpret_cast<float const*>(&mat.ks());
    float const* se = reinterpret_cast<float const*>(&mat.specular_exp());

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        for (int j = 0; j < spectrum<float>::num_samples; ++j)
        {
            float const* ca_j = reinterpret_cast<float const*>(&mat.ca()[j]);
            float const* cd_j = reinterpret_cast<float const*>(&mat.cd()[j]);
            float const* cs_j = reinterpret_cast<float const*>(&mat.cs()[j]);
            result[i].ca()[j] = ca_j[i];
            result[i].cd()[j] = cd_j[i];
            result[i].cs()[j] = cs_j[i];
        }
        result[i].ka() = ka[i];
        result[i].kd() = kd[i];
        result[i].ks() = ks[i];
        result[i].specular_exp() = se[i];
    }

    return result;
}

} // simd
} // visioanray
