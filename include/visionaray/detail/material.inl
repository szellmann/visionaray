// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

namespace visionaray
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack four materials into a single SIMD material
//

inline emissive<float4> pack(std::array<emissive<float>, 4> const& mats)
{
    emissive<float4> result;

    result.set_ce( pack(mats[0].get_ce(), mats[1].get_ce(), mats[2].get_ce(), mats[3].get_ce()) );
    result.set_ls( float4(mats[0].get_ls(), mats[1].get_ls(), mats[2].get_ls(), mats[3].get_ls()) );

    return result;
}


inline matte<float4> pack(std::array<matte<float>, 4> const& mats)
{
    matte<float4> result;

    result.set_ca( pack(mats[0].get_ca(), mats[1].get_ca(), mats[2].get_ca(), mats[3].get_ca()) );
    result.set_cd( pack(mats[0].get_cd(), mats[1].get_cd(), mats[2].get_cd(), mats[3].get_cd()) );
    result.set_ka( float4(mats[0].get_ka(), mats[1].get_ka(), mats[2].get_ka(), mats[3].get_ka()) );
    result.set_kd( float4(mats[0].get_kd(), mats[1].get_kd(), mats[2].get_kd(), mats[3].get_kd()) );

    return result;
}

inline mirror<float4> pack(std::array<mirror<float>, 4> const& mats)
{
    mirror<float4> result;

    result.set_cr( pack(mats[0].get_cr(), mats[1].get_cr(), mats[2].get_cr(), mats[3].get_cr()) );
    result.set_kr( float4(mats[0].get_kr(), mats[1].get_kr(), mats[2].get_kr(), mats[3].get_kr()) );

    return result;
}

inline plastic<float4> pack(std::array<plastic<float>, 4> const& mats)
{
    plastic<float4> result;

    result.set_ca( pack(mats[0].get_ca(), mats[1].get_ca(), mats[2].get_ca(), mats[3].get_ca()) );
    result.set_cd( pack(mats[0].get_cd(), mats[1].get_cd(), mats[2].get_cd(), mats[3].get_cd()) );
    result.set_cs( pack(mats[0].get_cs(), mats[1].get_cs(), mats[2].get_cs(), mats[3].get_cs()) );
    result.set_ka( float4(mats[0].get_ka(), mats[1].get_ka(), mats[2].get_ka(), mats[3].get_ka()) );
    result.set_kd( float4(mats[0].get_kd(), mats[1].get_kd(), mats[2].get_kd(), mats[3].get_kd()) );
    result.set_ks( float4(mats[0].get_ks(), mats[1].get_ks(), mats[2].get_ks(), mats[3].get_ks()) );
    result.set_specular_exp( float4(
            mats[0].get_specular_exp(),
            mats[1].get_specular_exp(),
            mats[2].get_specular_exp(),
            mats[3].get_specular_exp()
            ) );

    return result;
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Functions to pack eight materials into a single SIMD material
//

inline emissive<float8> pack(std::array<emissive<float>, 8> const& mats)
{
    emissive<float8> result;

    result.set_ce(pack(
            mats[0].get_ce(), mats[1].get_ce(), mats[2].get_ce(), mats[3].get_ce(),
            mats[4].get_ce(), mats[5].get_ce(), mats[6].get_ce(), mats[7].get_ce()
            ));

    result.set_ls(float8(
            mats[0].get_ls(), mats[1].get_ls(), mats[2].get_ls(), mats[3].get_ls(),
            mats[4].get_ls(), mats[5].get_ls(), mats[6].get_ls(), mats[7].get_ls()
            ));

    return result;
}

inline matte<float8> pack(std::array<matte<float>, 8> const& mats)
{
    matte<float8> result;

    result.set_ca(pack(
            mats[0].get_ca(), mats[1].get_ca(), mats[2].get_ca(), mats[3].get_ca(),
            mats[4].get_ca(), mats[5].get_ca(), mats[6].get_ca(), mats[7].get_ca()
            ));
    result.set_cd(pack(
            mats[0].get_cd(), mats[1].get_cd(), mats[2].get_cd(), mats[3].get_cd(),
            mats[4].get_cd(), mats[5].get_cd(), mats[6].get_cd(), mats[7].get_cd()
            ));
    result.set_ka(float8(
            mats[0].get_ka(), mats[1].get_ka(), mats[2].get_ka(), mats[3].get_ka(),
            mats[4].get_ka(), mats[5].get_ka(), mats[6].get_ka(), mats[7].get_ka()
            ));
    result.set_kd(float8(
            mats[0].get_kd(), mats[1].get_kd(), mats[2].get_kd(), mats[3].get_kd(),
            mats[4].get_kd(), mats[5].get_kd(), mats[6].get_kd(), mats[7].get_kd()
            ));

    return result;
}

inline mirror<float8> pack(std::array<mirror<float>, 8> const& mats)
{
    mirror<float8> result;

    result.set_cr(pack(
            mats[0].get_cr(), mats[1].get_cr(), mats[2].get_cr(), mats[3].get_cr(),
            mats[4].get_cr(), mats[5].get_cr(), mats[6].get_cr(), mats[7].get_cr()
            ));
    result.set_kr(float8(
            mats[0].get_kr(), mats[1].get_kr(), mats[2].get_kr(), mats[3].get_kr(),
            mats[4].get_kr(), mats[5].get_kr(), mats[6].get_kr(), mats[7].get_kr()
            ));

    return result;
}

inline plastic<float8> pack(std::array<plastic<float>, 8> const& mats)
{
    using C = spectrum<float>;

    C ca[8]                         = { mats[0].get_ca(), mats[1].get_ca(), mats[2].get_ca(), mats[3].get_ca(),
                                        mats[4].get_ca(), mats[5].get_ca(), mats[6].get_ca(), mats[7].get_ca() };
    C cd[8]                         = { mats[0].get_cd(), mats[1].get_cd(), mats[2].get_cd(), mats[3].get_cd(),
                                        mats[4].get_cd(), mats[5].get_cd(), mats[6].get_cd(), mats[7].get_cd() };
    C cs[8]                         = { mats[0].get_cs(), mats[1].get_cs(), mats[2].get_cs(), mats[3].get_cs(),
                                        mats[4].get_cs(), mats[5].get_cs(), mats[6].get_cs(), mats[7].get_cs() };
    VSNRAY_ALIGN(32) float ka[8]    = { mats[0].get_ka(), mats[1].get_ka(), mats[2].get_ka(), mats[3].get_ka(),
                                        mats[4].get_ka(), mats[5].get_ka(), mats[6].get_ka(), mats[7].get_ka() };
    VSNRAY_ALIGN(32) float kd[8]    = { mats[0].get_kd(), mats[1].get_kd(), mats[2].get_kd(), mats[3].get_kd(),
                                        mats[4].get_kd(), mats[5].get_kd(), mats[6].get_kd(), mats[7].get_kd() };
    VSNRAY_ALIGN(32) float ks[8]    = { mats[0].get_ks(), mats[1].get_ks(), mats[2].get_ks(), mats[3].get_ks(),
                                        mats[4].get_ks(), mats[5].get_ks(), mats[6].get_ks(), mats[7].get_ks() };
    VSNRAY_ALIGN(32) float exp[8]   = { mats[0].get_specular_exp(), mats[1].get_specular_exp(), mats[2].get_specular_exp(), mats[3].get_specular_exp(),
                                        mats[4].get_specular_exp(), mats[5].get_specular_exp(), mats[6].get_specular_exp(), mats[7].get_specular_exp() };

    plastic<float8> result;

    spectrum<float8> ca8;
    spectrum<float8> cd8;
    spectrum<float8> cs8;

    for (size_t d = 0; d < C::num_samples; ++d)
    {
        ca8[d] = float8(
                ca[0][d], ca[1][d], ca[2][d], ca[3][d],
                ca[4][d], ca[5][d], ca[6][d], ca[7][d]
                );

        cd8[d] = float8(
                cd[0][d], cd[1][d], cd[2][d], cd[3][d],
                cd[4][d], cd[5][d], cd[6][d], cd[7][d]
                );

        cs8[d] = float8(
                cs[0][d], cs[1][d], cs[2][d], cs[3][d],
                cs[4][d], cs[5][d], cs[6][d], cs[7][d]
                );
    }

    result.set_ca(ca8);
    result.set_cd(cd8);
    result.set_cs(cs8);
    result.set_ka(ka);
    result.set_kd(kd);
    result.set_ks(ks);
    result.set_specular_exp(exp);

    return result;
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd
} // visioanray
