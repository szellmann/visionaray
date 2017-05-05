// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vector>

#include <visionaray/array.h>
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
    pl.cd() = from_rgb(vec3(0.0f, 0.1f, 0.2f));
    pl.cs() = from_rgb(vec3(0.0f, 0.2f, 0.4f));
    pl.kd() = 1.0f;
    pl.ks() = 0.5f;

    mirror<float> mi;
    mi.cr() = from_rgb(vec3(1.0f, 1.0f, 1.0f));
    mi.kr() = 1.0f;
    mi.ior() = spectrum<float>(1.34f);
    mi.absorption() = spectrum<float>(0.0f);

    matte<float> ma;
    ma.cd() = from_rgb(vec3(1.0f, 0.0f, 0.0f));
    ma.kd() = 1.0f;

    emissive<float> em;
    em.ce() = from_rgb(vec3(3.0f, 3.0f, 3.0f));
    em.ls() = 5.0f;

    array<material_type, 4> mats{{
            material_type(pl),
            material_type(mi),
            material_type(ma),
            material_type(em)
            }};

    auto simd_material = simd::pack(mats);

    auto arr = simd::unpack(simd_material);

    auto m1 = apply_visitor( cast_visitor<plastic<float>>(),  arr[0] );
    auto m2 = apply_visitor( cast_visitor<mirror<float>>(),   arr[1] );
    auto m3 = apply_visitor( cast_visitor<matte<float>>(),    arr[2] );
    auto m4 = apply_visitor( cast_visitor<emissive<float>>(), arr[3] );

    // plastic

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m1.cd()[i], pl.cd()[i] );
        EXPECT_FLOAT_EQ( m1.cs()[i], pl.cs()[i] );
    }
    EXPECT_FLOAT_EQ( m1.kd(), pl.kd() );


    // mirror

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m2.cr()[i], mi.cr()[i] );
        EXPECT_FLOAT_EQ( m2.ior()[i], mi.ior()[i] );
        EXPECT_FLOAT_EQ( m2.absorption()[i], mi.absorption()[i] );
    }
    EXPECT_FLOAT_EQ( m2.kr(), mi.kr() );


    // matte

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m3.cd()[i], ma.cd()[i] );
    }
    EXPECT_FLOAT_EQ( m3.kd(), ma.kd() );


    // emissive

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( m4.ce()[i], em.ce()[i] );
    }
    EXPECT_FLOAT_EQ( m4.ls(), em.ls() );
}
