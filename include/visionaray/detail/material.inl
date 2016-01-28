// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

namespace visionaray
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack N materials into a single SIMD material
//

template <size_t N>
inline emissive<typename simd::float_from_simd_width<N>::type> pack(
        std::array<emissive<float>, N> const& mats
        )
{
    using T = typename simd::float_from_simd_width<N>::type;
    using float_array = typename simd::aligned_array<T>::type;

    emissive<T> result;

    std::array<spectrum<float>, N>  ce;
    float_array                     ls;

    for (size_t i = 0; i < N; ++i)
    {
        ce[i] = mats[i].get_ce();
        ls[i] = mats[i].get_ls();
    }

    result.set_ce( pack(ce) );
    result.set_ls( T(ls) );

    return result;
}

template <size_t N>
inline matte<typename simd::float_from_simd_width<N>::type> pack(
        std::array<matte<float>, N> const& mats
        )
{
    using T = typename simd::float_from_simd_width<N>::type;
    using float_array = typename simd::aligned_array<T>::type;

    matte<T> result;

    std::array<spectrum<float>, N>  ca;
    std::array<spectrum<float>, N>  cd;
    float_array                     ka;
    float_array                     kd;

    for (size_t i = 0; i < N; ++i)
    {
        ca[i] = mats[i].get_ca();
        cd[i] = mats[i].get_cd();
        ka[i] = mats[i].get_ka();
        kd[i] = mats[i].get_kd();
    }

    result.set_ca( pack(ca) );
    result.set_cd( pack(cd) );
    result.set_ka( T(ka) );
    result.set_kd( T(kd) );

    return result;
}

template <size_t N>
inline mirror<typename simd::float_from_simd_width<N>::type> pack(
        std::array<mirror<float>, N> const& mats
        )
{
    using T = typename simd::float_from_simd_width<N>::type;
    using float_array = typename simd::aligned_array<T>::type;

    mirror<T> result;

    std::array<spectrum<float>, N>  cr;
    float_array                     kr;

    for (size_t i = 0; i < N; ++i)
    {
        cr[i] = mats[i].get_cr();
        kr[i] = mats[i].get_kr();
    }

    result.set_cr( pack(cr) );
    result.set_kr( T(kr) );

    return result;
}

template <size_t N>
inline plastic<typename simd::float_from_simd_width<N>::type> pack(
    std::array<plastic<float>, N> const& mats
    )
{
    using T = typename simd::float_from_simd_width<N>::type;
    using float_array = typename simd::aligned_array<T>::type;

    plastic<T> result;

    std::array<spectrum<float>, N>  ca;
    std::array<spectrum<float>, N>  cd;
    std::array<spectrum<float>, N>  cs;
    float_array                     ka;
    float_array                     kd;
    float_array                     ks;
    float_array                     se;

    for (size_t i = 0; i < N; ++i)
    {
        ca[i] = mats[i].get_ca();
        cd[i] = mats[i].get_cd();
        cs[i] = mats[i].get_cs();
        ka[i] = mats[i].get_ka();
        kd[i] = mats[i].get_kd();
        ks[i] = mats[i].get_ks();
        se[i] = mats[i].get_specular_exp();
    }

    result.set_ca( pack(ca) );
    result.set_cd( pack(cd) );
    result.set_cs( pack(cs) );
    result.set_ka( T(ka) );
    result.set_kd( T(kd) );
    result.set_ks( T(ks) );
    result.set_specular_exp( T(se) );

    return result;
}

} // simd
} // visioanray
