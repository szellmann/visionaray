// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Functions to pack four materials into a single SIMD material
//

inline phong<simd::float4> pack
(
    phong<float> const& m1, phong<float> const& m2,
    phong<float> const& m3, phong<float> const& m4
)
{
    vector<3, float> ca[4]          = { m1.get_ca(), m2.get_ca(), m3.get_ca(), m4.get_ca() };
    vector<3, float> cd[4]          = { m1.get_cd(), m2.get_cd(), m3.get_cd(), m4.get_cd() };
    VSNRAY_ALIGN(16) float ka[4]    = { m1.get_ka(), m2.get_ka(), m3.get_ka(), m4.get_ka() };
    VSNRAY_ALIGN(16) float kd[4]    = { m1.get_kd(), m2.get_kd(), m3.get_kd(), m4.get_kd() };
    VSNRAY_ALIGN(16) float ks[4]    = { m1.get_ks(), m2.get_ks(), m3.get_ks(), m4.get_ks() };
    VSNRAY_ALIGN(16) float exp[4]   = { m1.get_specular_exp(), m2.get_specular_exp(), m3.get_specular_exp(), m4.get_specular_exp() };

    phong<simd::float4> result;
    result.set_ca
    (
        vector<3, simd::float4>
        (
            simd::float4( ca[0].x, ca[1].x, ca[2].x, ca[3].x ),
            simd::float4( ca[0].y, ca[1].y, ca[2].y, ca[3].y ),
            simd::float4( ca[0].z, ca[1].z, ca[2].z, ca[3].z )
        )
    );
    result.set_cd
    (
        vector<3, simd::float4>
        (
            simd::float4( cd[0].x, cd[1].x, cd[2].x, cd[3].x ),
            simd::float4( cd[0].y, cd[1].y, cd[2].y, cd[3].y ),
            simd::float4( cd[0].z, cd[1].z, cd[2].z, cd[3].z )
        )
    );
    result.set_ka(ka);
    result.set_kd(kd);
    result.set_ks(ks);
    result.set_specular_exp(exp);
    return result;
}

inline emissive<simd::float4> pack
(
    emissive<float> const& m1, emissive<float> const& m2,
    emissive<float> const& m3, emissive<float> const& m4
)
{
    vector<3, float> ce[4]          = { m1.get_ce(), m2.get_ce(), m3.get_ce(), m4.get_ce() };
    VSNRAY_ALIGN(16) float ls[4]    = { m1.get_ls(), m2.get_ls(), m3.get_ls(), m4.get_ls() };

    emissive<simd::float4> result;
    result.set_ce
    (
        vector<3, simd::float4>
        (
            simd::float4( ce[0].x, ce[1].x, ce[2].x, ce[3].x ),
            simd::float4( ce[0].y, ce[1].y, ce[2].y, ce[3].y ),
            simd::float4( ce[0].z, ce[1].z, ce[2].z, ce[3].z )
        )
    );
    result.set_ls(ls);

    return result;
}

inline generic_mat<simd::float4> pack
(
    generic_mat<float> const& m1, generic_mat<float> const& m2,
    generic_mat<float> const& m3, generic_mat<float> const& m4
)
{
    return generic_mat<simd::float4>(m1, m2, m3, m4);
}


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// Functions to pack eight materials into a single SIMD material
//

inline phong<simd::float8> pack
(
    phong<float> const& m1, phong<float> const& m2, phong<float> const& m3, phong<float> const& m4,
    phong<float> const& m5, phong<float> const& m6, phong<float> const& m7, phong<float> const& m8
)
{
    vector<3, float> cd[8]          = { m1.get_cd(), m2.get_cd(), m3.get_cd(), m4.get_cd(),
                                        m5.get_cd(), m6.get_cd(), m7.get_cd(), m8.get_cd() };
    vector<3, float> ca[8]          = { m1.get_ca(), m2.get_ca(), m3.get_ca(), m4.get_ca(),
                                        m5.get_ca(), m6.get_ca(), m7.get_ca(), m8.get_ca() };
    VSNRAY_ALIGN(32) float ka[8]    = { m1.get_ka(), m2.get_ka(), m3.get_ka(), m4.get_ka(),
                                        m5.get_ka(), m6.get_ka(), m7.get_ka(), m8.get_ka() };
    VSNRAY_ALIGN(32) float kd[8]    = { m1.get_kd(), m2.get_kd(), m3.get_kd(), m4.get_kd(),
                                        m5.get_kd(), m6.get_kd(), m7.get_kd(), m8.get_kd() };
    VSNRAY_ALIGN(32) float ks[8]    = { m1.get_ks(), m2.get_ks(), m3.get_ks(), m4.get_ks(),
                                        m5.get_ks(), m6.get_ks(), m7.get_ks(), m8.get_ks() };
    VSNRAY_ALIGN(32) float exp[8]   = { m1.get_specular_exp(), m2.get_specular_exp(), m3.get_specular_exp(), m4.get_specular_exp(),
                                        m5.get_specular_exp(), m6.get_specular_exp(), m7.get_specular_exp(), m8.get_specular_exp() };

    phong<simd::float8> result;
    result.set_ca
    (
        vector<3, simd::float8>
        (
            simd::float8( ca[0].x, ca[1].x, ca[2].x, ca[3].x, ca[4].x, ca[5].x, ca[6].x, ca[7].x ),
            simd::float8( ca[0].y, ca[1].y, ca[2].y, ca[3].y, ca[4].y, ca[5].y, ca[6].y, ca[7].y ),
            simd::float8( ca[0].z, ca[1].z, ca[2].z, ca[3].z, ca[4].z, ca[5].z, ca[6].z, ca[7].z )
        )
    );
    result.set_cd
    (
        vector<3, simd::float8>
        (
            simd::float8( cd[0].x, cd[1].x, cd[2].x, cd[3].x, cd[4].x, cd[5].x, cd[6].x, cd[7].x ),
            simd::float8( cd[0].y, cd[1].y, cd[2].y, cd[3].y, cd[4].y, cd[5].y, cd[6].y, cd[7].y ),
            simd::float8( cd[0].z, cd[1].z, cd[2].z, cd[3].z, cd[4].z, cd[5].z, cd[6].z, cd[7].z )
        )
    );
    result.set_ka(ka);
    result.set_kd(kd);
    result.set_ks(ks);
    result.set_specular_exp(exp);

    return result;
}


#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


} // visioanray
