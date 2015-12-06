// This file is distributed under the MIT license.
// See the LICENSE file for details.
#include <iostream>
#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
//
// Test setup:
//
//
//      Y
//
//      -
//
//  300 -               B--------
//                      |   C-- |----
//  250 -               |   |   |   |
//                      ---------   |
//  200 -                   |       |
//                          ---------
//  150 -
//
//  100 -   A--------   D---F----
//          | G---- |   |   |   |
//   50 -   | |   | |   |   E----
//          | ----- |   |   |   |
//    0 -   ---------   ---------
//
//      /   |   |   |   |   |   |   |   |   X
//          0  50  100 150 200 250 300
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Helper functions
//


template <typename RectA, typename RectB>
RectA convert_xywh(RectB const& rect)
{
    return RectA(rect.x, rect.y, rect.w, rect.h);
}


//-------------------------------------------------------------------------------------------------
// Tests
//

TEST(Rect, XYWH)
{

    // init -----------------------------------------------

    recti Ai(  0,   0, 100, 100);
    recti Bi(150, 225, 100,  75);
    recti Ci(200, 175, 100, 100);
    recti Di(150,   0,  50, 100);
    recti Ei(200,   0,  50,  50);
    recti Fi(200,  50,  50,  50);
    recti Gi( 25,  25,  50,  50);

    rectf Af = convert_xywh<rectf>(Ai);
    rectf Bf = convert_xywh<rectf>(Bi);
    rectf Cf = convert_xywh<rectf>(Ci);
    rectf Df = convert_xywh<rectf>(Di);
    rectf Ef = convert_xywh<rectf>(Ei);
    rectf Ff = convert_xywh<rectf>(Fi);


    // test validity checks -------------------------------

    recti empty(0, 0, 0, 0);
    recti valid = Ai;
    recti invalid;
    invalid.invalidate();

    EXPECT_TRUE( empty.valid());
    EXPECT_TRUE( empty.empty());
    EXPECT_FALSE(empty.invalid());

    EXPECT_TRUE( valid.valid());
    EXPECT_FALSE(valid.empty());
    EXPECT_FALSE(valid.invalid());

    EXPECT_FALSE(invalid.valid());
    EXPECT_TRUE( invalid.empty());
    EXPECT_TRUE( invalid.invalid());

    invalid = Ai;
    invalid.w = -1;

    EXPECT_FALSE(invalid.valid());
    EXPECT_TRUE( invalid.empty());
    EXPECT_TRUE( invalid.invalid());

    invalid = Ai;
    invalid.h = -1;

    EXPECT_FALSE(invalid.valid());
    EXPECT_TRUE( invalid.empty());
    EXPECT_TRUE( invalid.invalid());


    // test contains() ------------------------------------

    // contains point?
    EXPECT_FALSE(Ai.contains(vec2i(-1, -1)));
    EXPECT_TRUE( Ai.contains(vec2i(0, 0)));
    EXPECT_TRUE( Ai.contains(vec2i(1, 1)));
    EXPECT_TRUE( Ai.contains(vec2i(99, 99)));
    EXPECT_TRUE( Ai.contains(vec2i(100, 100)));
    EXPECT_FALSE(Ai.contains(vec2i(101, 101)));

    // contains rectangle?
    EXPECT_TRUE( Ai.contains(Ai));
    EXPECT_TRUE( Ai.contains(Gi));
    EXPECT_FALSE(Gi.contains(Ai));
    EXPECT_FALSE(Ai.contains(Bi));
    EXPECT_FALSE(Bi.contains(Ci));
    EXPECT_FALSE(Ci.contains(Bi));
    EXPECT_FALSE(Di.contains(Ei));
    EXPECT_FALSE(Ei.contains(Fi));


    // test overlapping() ---------------------------------

    EXPECT_FALSE(overlapping(Ai, Bi));
    EXPECT_TRUE( overlapping(Bi, Ci));
    EXPECT_TRUE( overlapping(Di, Ei));
    EXPECT_TRUE( overlapping(Ei, Fi));

    EXPECT_FALSE(overlapping(Af, Bf));
    EXPECT_TRUE( overlapping(Bf, Cf));
    EXPECT_TRUE( overlapping(Df, Ef));
    EXPECT_TRUE( overlapping(Ef, Ff));


    // test combine() -------------------------------------

    EXPECT_TRUE(combine(Ei, Fi) == recti(200, 0, 50, 100));
    EXPECT_TRUE(combine(Bi, Ci) == recti(150, 175, 150, 125));
    EXPECT_TRUE(combine(Di, combine(Ei, Fi)) == recti(150, 0, 100, 100));


    // test intersect() -----------------------------------

    EXPECT_TRUE(intersect(Ai, Bi).empty());
    EXPECT_TRUE(intersect(Di, Ei).empty());
    EXPECT_TRUE(intersect(Ei, Fi).empty());
    EXPECT_TRUE(intersect(Bi, Ci) == recti(200, 225, 50, 50));

}
