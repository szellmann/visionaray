// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// Functions to pack four materials into a single SIMD material
//

inline plastic<float4> pack(
        plastic<float> const& m1,
        plastic<float> const& m2,
        plastic<float> const& m3,
        plastic<float> const& m4
        )
{
    plastic<float4> result;

    result.set_ca( pack(m1.get_ca(), m2.get_ca(), m3.get_ca(), m4.get_ca()) );
    result.set_cd( pack(m1.get_cd(), m2.get_cd(), m3.get_cd(), m4.get_cd()) );
    result.set_cs( pack(m1.get_cs(), m2.get_cs(), m3.get_cs(), m4.get_cs()) );
    result.set_ka( float4(m1.get_ka(), m2.get_ka(), m3.get_ka(), m4.get_ka()) );
    result.set_kd( float4(m1.get_kd(), m2.get_kd(), m3.get_kd(), m4.get_kd()) );
    result.set_ks( float4(m1.get_ks(), m2.get_ks(), m3.get_ks(), m4.get_ks()) );
    result.set_specular_exp( float4(m1.get_specular_exp(), m2.get_specular_exp(), m3.get_specular_exp(), m4.get_specular_exp()) );

    return result;
}

inline emissive<float4> pack(
        emissive<float> const& m1,
        emissive<float> const& m2,
        emissive<float> const& m3,
        emissive<float> const& m4
        )
{
    emissive<float4> result;

    result.set_ce(
            pack(m1.get_ce(),
            m2.get_ce(),
            m3.get_ce(),
            m4.get_ce())
            );

    result.set_ls( float4(
            m1.get_ls(),
            m2.get_ls(),
            m3.get_ls(),
            m4.get_ls())
            );

    return result;
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Functions to pack eight materials into a single SIMD material
//

inline plastic<float8> pack(
        plastic<float> const& m1,
        plastic<float> const& m2,
        plastic<float> const& m3,
        plastic<float> const& m4,
        plastic<float> const& m5,
        plastic<float> const& m6,
        plastic<float> const& m7,
        plastic<float> const& m8
        )
{
    using C = spectrum<float>;

    C ca[8]                         = { m1.get_ca(), m2.get_ca(), m3.get_ca(), m4.get_ca(),
                                        m5.get_ca(), m6.get_ca(), m7.get_ca(), m8.get_ca() };
    C cd[8]                         = { m1.get_cd(), m2.get_cd(), m3.get_cd(), m4.get_cd(),
                                        m5.get_cd(), m6.get_cd(), m7.get_cd(), m8.get_cd() };
    C cs[8]                         = { m1.get_cs(), m2.get_cs(), m3.get_cs(), m4.get_cs(),
                                        m5.get_cs(), m6.get_cs(), m7.get_cs(), m8.get_cs() };
    VSNRAY_ALIGN(32) float ka[8]    = { m1.get_ka(), m2.get_ka(), m3.get_ka(), m4.get_ka(),
                                        m5.get_ka(), m6.get_ka(), m7.get_ka(), m8.get_ka() };
    VSNRAY_ALIGN(32) float kd[8]    = { m1.get_kd(), m2.get_kd(), m3.get_kd(), m4.get_kd(),
                                        m5.get_kd(), m6.get_kd(), m7.get_kd(), m8.get_kd() };
    VSNRAY_ALIGN(32) float ks[8]    = { m1.get_ks(), m2.get_ks(), m3.get_ks(), m4.get_ks(),
                                        m5.get_ks(), m6.get_ks(), m7.get_ks(), m8.get_ks() };
    VSNRAY_ALIGN(32) float exp[8]   = { m1.get_specular_exp(), m2.get_specular_exp(), m3.get_specular_exp(), m4.get_specular_exp(),
                                        m5.get_specular_exp(), m6.get_specular_exp(), m7.get_specular_exp(), m8.get_specular_exp() };

    plastic<float8> result;

    spectrum<float8> ca8;
    spectrum<float8> cd8;
    spectrum<float8> cs8;

    for (size_t d = 0; d < C::num_samples; ++d)
    {
        ca8[d] = float8(
                ca[0][d],
                ca[1][d],
                ca[2][d],
                ca[3][d],
                ca[4][d],
                ca[5][d],
                ca[6][d],
                ca[7][d]
                );

        cd8[d] = float8(
                cd[0][d],
                cd[1][d],
                cd[2][d],
                cd[3][d],
                cd[4][d],
                cd[5][d],
                cd[6][d],
                cd[7][d]
                );

        cs8[d] = float8(
                cs[0][d],
                cs[1][d],
                cs[2][d],
                cs[3][d],
                cs[4][d],
                cs[5][d],
                cs[6][d],
                cs[7][d]
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
