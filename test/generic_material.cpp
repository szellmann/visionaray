// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vector>

#include <visionaray/generic_material.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Visitors
//

template <typename Material>
struct cast_visitor
{
    using return_type = Material;

    template <typename X>
    return_type operator()(X) const
    {
        return return_type();
    }

    return_type operator()(Material const& ref) const
    {
        return ref;
    }
};


//-------------------------------------------------------------------------------------------------
// Test simd::(un)pack()
//

TEST(GenericMaterial, SIMD)
{
    using material_type = generic_material<
        plastic<float>,
        mirror<float>,
        matte<float>,
        emissive<float>
        >;

    plastic<float> pl;
    pl.set_cd( from_rgb(vec3(0.0f, 0.1f, 0.2f)) );
    pl.set_cs( from_rgb(vec3(0.0f, 0.2f, 0.4f)) );
    pl.set_kd( 1.0f );
    pl.set_ks( 0.5f );

    mirror<float> mi;
    mi.set_cr( from_rgb(vec3(1.0f, 1.0f, 1.0f)) );
    mi.set_kr( 1.0f );
    mi.set_ior( 1.34f );
    mi.set_absorption( 0.0f );

    matte<float> ma;
    ma.set_cd( from_rgb(vec3(1.0f, 0.0f, 0.0f)) );
    ma.set_kd( 1.0f );

    emissive<float> em;
    em.set_ce( from_rgb(vec3(3.0f, 3.0f, 3.0f)) );
    em.set_ls( 5.0f );

    auto simd_material = simd::pack(
            material_type(pl),
            material_type(mi),
            material_type(ma),
            material_type(em)
            );

    auto arr = simd::unpack(simd_material);

    auto m1 = apply_visitor( cast_visitor<plastic<float>>(),  arr[0] );
    auto m2 = apply_visitor( cast_visitor<mirror<float>>(),   arr[1] );
    auto m3 = apply_visitor( cast_visitor<matte<float>>(),    arr[2] );
    auto m4 = apply_visitor( cast_visitor<emissive<float>>(), arr[3] );

    // plastic

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m1.get_cd()[i], pl.get_cd()[i] );
        EXPECT_FLOAT_EQ( m1.get_cs()[i], pl.get_cs()[i] );
    }
    EXPECT_FLOAT_EQ( m1.get_kd(), pl.get_kd() );


    // mirror

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m2.get_cr()[i], mi.get_cr()[i] );
        EXPECT_FLOAT_EQ( m2.get_ior()[i], mi.get_ior()[i] );
        EXPECT_FLOAT_EQ( m2.get_absorption()[i], mi.get_absorption()[i] );
    }
    EXPECT_FLOAT_EQ( m2.get_kr(), mi.get_kr() );


    // matte

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m3.get_cd()[i], ma.get_cd()[i] );
    }
    EXPECT_FLOAT_EQ( m3.get_kd(), ma.get_kd() );


    // emissive

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m4.get_ce()[i], em.get_ce()[i] );
    }
    EXPECT_FLOAT_EQ( m4.get_ls(), em.get_ls() );
}
