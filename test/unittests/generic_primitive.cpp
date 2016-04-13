// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>
#include <visionaray/bvh.h>
#include <visionaray/generic_primitive.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Very simple custom primitive type for testing
//

struct custom_primitive {};

aabb get_bounds(custom_primitive)
{
    return aabb( { -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f } );
}


//-------------------------------------------------------------------------------------------------
// Test get_bounds()
//

TEST(GenericPrimitive, GetBounds)
{
    // sphere ---------------------------------------------

    using prim1_t = generic_primitive<basic_sphere<float>>;

    basic_sphere<float> s;
    s.center = vec3(0.0f, 0.5f, 1.0f);
    s.radius = 1.0f;

    prim1_t p1(s);

    EXPECT_TRUE( get_bounds(p1) == get_bounds(s) );


    // sphere and triangle --------------------------------

    using prim2_t = generic_primitive<basic_sphere<float>, basic_triangle<3, float>>;

    basic_triangle<3, float> t;
    t.v1 = vec3(0.0f, 0.25f, 0.5f);
    t.e1 = vec3(0.0f, 0.0f, 0.0f) - t.v1;
    t.e2 = vec3(0.0f, 1.0f, 0.0f) - t.v1;

    prim2_t ps2(s);
    prim2_t pt2(t);

    EXPECT_TRUE( get_bounds(ps2) == get_bounds(s) );
    EXPECT_TRUE( get_bounds(pt2) == get_bounds(t) );


    // custom prim ----------------------------------------

    using prim3_t = generic_primitive<custom_primitive>;

    custom_primitive c;

    prim3_t p3(c);

    EXPECT_TRUE( get_bounds(p3) == get_bounds(c) );
}


//-------------------------------------------------------------------------------------------------
// Test split_primitive()
//

TEST(GenericPrimitive, SplitPrimitive)
{
    // sphere ---------------------------------------------

    using prim1_t = generic_primitive<basic_sphere<float>>;

    basic_sphere<float> s;
    s.center = vec3(0.0f, 0.5f, 1.0f);
    s.radius = 1.0f;

    prim1_t p1(s);

    aabb Ls, Rs, Lp, Rp;


    // X

    split_primitive(Ls, Rs, 0.0f, cartesian_axis<3>::X, s);
    split_primitive(Lp, Rp, 0.0f, cartesian_axis<3>::X, p1);

    EXPECT_TRUE( Ls == Lp );
    EXPECT_TRUE( Rs == Rp );

    // Y

    split_primitive(Ls, Rs, 0.25f, cartesian_axis<3>::Y, s);
    split_primitive(Lp, Rp, 0.25f, cartesian_axis<3>::Y, p1);

    EXPECT_TRUE( Ls == Lp );
    EXPECT_TRUE( Rs == Rp );

    // Z

    split_primitive(Ls, Rs, 0.5f, cartesian_axis<3>::Z, s);
    split_primitive(Lp, Rp, 0.5f, cartesian_axis<3>::Z, p1);

    EXPECT_TRUE( Ls == Lp );
    EXPECT_TRUE( Rs == Rp );

    // L empty, R not empty

    split_primitive(Lp, Rp, -1.0f, cartesian_axis<3>::X, p1);

    EXPECT_TRUE( Lp.empty() );
    EXPECT_FALSE( Rp.empty() );


    // sphere and triangle --------------------------------

    using prim2_t = generic_primitive<basic_sphere<float>, basic_triangle<3, float>>;

    basic_triangle<3, float> t;
    t.v1 = vec3(0.0f, 0.25f, 0.5f);
    t.e1 = vec3(0.0f, 0.0f, 0.0f) - t.v1;
    t.e2 = vec3(0.0f, 1.0f, 0.0f) - t.v1;

    prim2_t p2(t);

    aabb Lt, Rt;

    // some basic tests

    split_primitive(Lt, Rt, 0.5f, cartesian_axis<3>::Y, t);
    split_primitive(Lp, Rp, 0.5f, cartesian_axis<3>::Y, p2);

    EXPECT_TRUE( Lt == Lp );
    EXPECT_TRUE( Rt == Rp );
}
