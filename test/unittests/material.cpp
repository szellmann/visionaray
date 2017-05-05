// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vector>

#include <visionaray/array.h>
#include <visionaray/generic_material.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test simd::(un)pack()
//

TEST(Material, SIMD)
{
    plastic<float> pl0;
    pl0.cd() = from_rgb(vec3(0.0f, 0.1f, 0.2f));
    pl0.cs() = from_rgb(vec3(0.0f, 0.2f, 0.4f));
    pl0.kd() = 1.0f;
    pl0.ks() = 0.5f;

    plastic<float> pl1;
    pl1.cd() = from_rgb(vec3(0.1f, 0.5f, 0.9f));
    pl1.cs() = from_rgb(vec3(0.5f, 0.7f, 0.9f));
    pl1.kd() = 1.1f;
    pl1.ks() = 0.6f;

    plastic<float> pl2;
    pl2.cd() = from_rgb(vec3(1.0f, 1.1f, 1.2f));
    pl2.cs() = from_rgb(vec3(1.0f, 1.2f, 1.4f));
    pl2.kd() = 1.2f;
    pl2.ks() = 0.7f;

    plastic<float> pl3;
    pl3.cd() = from_rgb(vec3(0.0f, 0.0f, 0.0f));
    pl3.cs() = from_rgb(vec3(0.0f, 0.0f, 0.0f));
    pl3.kd() = 1.3f;
    pl3.ks() = 0.8f;


    mirror<float> mi0;
    mi0.cr() = from_rgb(vec3(1.0f, 1.0f, 1.0f));
    mi0.kr() = 1.0f;
    mi0.ior() = spectrum<float>(1.34f);
    mi0.absorption() = spectrum<float>(0.0f);

    mirror<float> mi1;
    mi1.cr() = from_rgb(vec3(1.0f, 1.0f, 0.0f));
    mi1.kr() = 0.75f;
    mi1.ior() = spectrum<float>(3.14f);
    mi1.absorption() = spectrum<float>(0.6f);

    mirror<float> mi2;
    mi2.cr() = from_rgb(vec3(0.0f, 1.0f, 1.0f));
    mi2.kr() = 0.5f;
    mi2.ior() = spectrum<float>(2.34f);
    mi2.absorption() = spectrum<float>(0.5f);

    mirror<float> mi3;
    mi3.cr() = from_rgb(vec3(1.0f, 0.0f, 1.0f));
    mi3.kr() = 0.25f;
    mi3.ior() = spectrum<float>(3.45f);
    mi3.absorption() = spectrum<float>(1.0f);


    matte<float> ma0;
    ma0.cd() = from_rgb(vec3(1.0f, 0.0f, 0.0f));
    ma0.kd() = 1.0f;

    matte<float> ma1;
    ma1.cd() = from_rgb(vec3(0.0f, 1.0f, 0.0f));
    ma1.kd() = 0.8f;

    matte<float> ma2;
    ma2.cd() = from_rgb(vec3(0.0f, 0.0f, 1.0f));
    ma2.kd() = 0.6f;

    matte<float> ma3;
    ma3.cd() = from_rgb(vec3(1.0f, 0.0f, 1.0f));
    ma3.kd() = 0.4f;


    emissive<float> em0;
    em0.ce() = from_rgb(vec3(3.0f, 3.0f, 3.0f));
    em0.ls() = 5.0f;

    emissive<float> em1;
    em1.ce() = from_rgb(vec3(3.1f, 3.1f, 3.1f));
    em1.ls() = 6.0f;

    emissive<float> em2;
    em2.ce() = from_rgb(vec3(3.14f, 3.14f, 3.14f));
    em2.ls() = 7.0f;

    emissive<float> em3;
    em3.ce() = from_rgb(vec3(3.141f, 3.141f, 3.141f));
    em3.ls() = 8.0f;


    // pack

    array<plastic<float>, 4> pls{{ pl0, pl1, pl2, pl3 }};
    auto pll4 = simd::pack(pls);

    array<mirror<float>, 4> mis{{ mi0, mi1, mi2, mi3 }};
    auto mii4 = simd::pack(mis);

    array<matte<float>, 4> mas{{ ma0, ma1, ma2, ma3 }};
    auto maa4 = simd::pack(mas);

    array<emissive<float>, 4> ems{{ em0, em1, em2, em3 }};
    auto emm4 = simd::pack(ems);


    // unpack

    auto pll = unpack(pll4);
    auto mii = unpack(mii4);
    auto maa = unpack(maa4);
    auto emm = unpack(emm4);


    // plastic

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( pll[0].cd()[i], pl0.cd()[i] );
        EXPECT_FLOAT_EQ( pll[1].cd()[i], pl1.cd()[i] );
        EXPECT_FLOAT_EQ( pll[2].cd()[i], pl2.cd()[i] );
        EXPECT_FLOAT_EQ( pll[3].cd()[i], pl3.cd()[i] );
        EXPECT_FLOAT_EQ( pll[0].cs()[i], pl0.cs()[i] );
        EXPECT_FLOAT_EQ( pll[1].cs()[i], pl1.cs()[i] );
        EXPECT_FLOAT_EQ( pll[2].cs()[i], pl2.cs()[i] );
        EXPECT_FLOAT_EQ( pll[3].cs()[i], pl3.cs()[i] );
    }
    EXPECT_FLOAT_EQ( pll[0].kd(), pl0.kd() );
    EXPECT_FLOAT_EQ( pll[1].kd(), pl1.kd() );
    EXPECT_FLOAT_EQ( pll[2].kd(), pl2.kd() );
    EXPECT_FLOAT_EQ( pll[3].kd(), pl3.kd() );
    EXPECT_FLOAT_EQ( pll[0].ks(), pl0.ks() );
    EXPECT_FLOAT_EQ( pll[1].ks(), pl1.ks() );
    EXPECT_FLOAT_EQ( pll[2].ks(), pl2.ks() );
    EXPECT_FLOAT_EQ( pll[3].ks(), pl3.ks() );


    // mirror

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( mii[0].cr()[i], mi0.cr()[i] );
        EXPECT_FLOAT_EQ( mii[1].cr()[i], mi1.cr()[i] );
        EXPECT_FLOAT_EQ( mii[2].cr()[i], mi2.cr()[i] );
        EXPECT_FLOAT_EQ( mii[3].cr()[i], mi3.cr()[i] );
//      EXPECT_FLOAT_EQ( mii[0].ior()[i], mi0.ior()[i] );
//      EXPECT_FLOAT_EQ( mii[1].ior()[i], mi1.ior()[i] );
//      EXPECT_FLOAT_EQ( mii[2].ior()[i], mi2.ior()[i] );
//      EXPECT_FLOAT_EQ( mii[3].ior()[i], mi3.ior()[i] );
//      EXPECT_FLOAT_EQ( mii[0].absorption()[i], mi0.absorption()[i] );
//      EXPECT_FLOAT_EQ( mii[1].absorption()[i], mi1.absorption()[i] );
//      EXPECT_FLOAT_EQ( mii[2].absorption()[i], mi2.absorption()[i] );
//      EXPECT_FLOAT_EQ( mii[3].absorption()[i], mi3.absorption()[i] );
    }
    EXPECT_FLOAT_EQ( mii[0].kr(), mi0.kr() );
    EXPECT_FLOAT_EQ( mii[1].kr(), mi1.kr() );
    EXPECT_FLOAT_EQ( mii[2].kr(), mi2.kr() );
    EXPECT_FLOAT_EQ( mii[3].kr(), mi3.kr() );


    // matte

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( maa[0].cd()[i], ma0.cd()[i] );
        EXPECT_FLOAT_EQ( maa[1].cd()[i], ma1.cd()[i] );
        EXPECT_FLOAT_EQ( maa[2].cd()[i], ma2.cd()[i] );
        EXPECT_FLOAT_EQ( maa[3].cd()[i], ma3.cd()[i] );
    }
    EXPECT_FLOAT_EQ( maa[0].kd(), ma0.kd() );
    EXPECT_FLOAT_EQ( maa[1].kd(), ma1.kd() );
    EXPECT_FLOAT_EQ( maa[2].kd(), ma2.kd() );
    EXPECT_FLOAT_EQ( maa[3].kd(), ma3.kd() );


    // emissive

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ( emm[0].ce()[i], em0.ce()[i] );
        EXPECT_FLOAT_EQ( emm[1].ce()[i], em1.ce()[i] );
        EXPECT_FLOAT_EQ( emm[2].ce()[i], em2.ce()[i] );
        EXPECT_FLOAT_EQ( emm[3].ce()[i], em3.ce()[i] );
    }
    EXPECT_FLOAT_EQ( emm[0].ls(), em0.ls() );
    EXPECT_FLOAT_EQ( emm[1].ls(), em1.ls() );
    EXPECT_FLOAT_EQ( emm[2].ls(), em2.ls() );
    EXPECT_FLOAT_EQ( emm[3].ls(), em3.ls() );
}
