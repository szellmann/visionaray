// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>
#include <limits>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// SIMD float
//

TEST(SIMD, SelectFloat)
{
    // float4

    {
        simd::mask4  m(1,1,0,0);
        simd::float4 a(0.0f, 1.0f, 2.0f, 3.0f);
        simd::float4 b(3.0f, 2.0f, 1.0f, 0.0f);

        simd::float4 c = select(m, a, b);
        EXPECT_FLOAT_EQ( simd::get<0>(c), 0.0f );
        EXPECT_FLOAT_EQ( simd::get<1>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get<2>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get<3>(c), 0.0f );
    }


    // float8

    {
        simd::mask8  m(1,1,1,1, 0,0,0,0);
        simd::float8 a(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
        simd::float8 b(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

        simd::float8 c = select(m, a, b);
        EXPECT_FLOAT_EQ( simd::get<0>(c), 0.0f );
        EXPECT_FLOAT_EQ( simd::get<1>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get<2>(c), 2.0f );
        EXPECT_FLOAT_EQ( simd::get<3>(c), 3.0f );
        EXPECT_FLOAT_EQ( simd::get<4>(c), 3.0f );
        EXPECT_FLOAT_EQ( simd::get<5>(c), 2.0f );
        EXPECT_FLOAT_EQ( simd::get<6>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get<7>(c), 0.0f );
    }


    // float16

    {
        simd::mask16  m(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
        simd::float16 a(
                 0.0f,  1.0f,  2.0f,  3.0f,
                 4.0f,  5.0f,  6.0f,  7.0f,
                 8.0f,  9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f
                );
        simd::float16 b(
                15.0f, 14.0f, 13.0f, 12.0f,
                11.0f, 10.0f,  9.0f,  8.0f,
                 7.0f,  6.0f,  5.0f,  4.0f,
                 3.0f,  2.0f,  1.0f,  0.0f
                );

        simd::float16 c = select(m, a, b);
        EXPECT_FLOAT_EQ( simd::get< 0>(c), 0.0f );
        EXPECT_FLOAT_EQ( simd::get< 1>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get< 2>(c), 2.0f );
        EXPECT_FLOAT_EQ( simd::get< 3>(c), 3.0f );
        EXPECT_FLOAT_EQ( simd::get< 4>(c), 4.0f );
        EXPECT_FLOAT_EQ( simd::get< 5>(c), 5.0f );
        EXPECT_FLOAT_EQ( simd::get< 6>(c), 6.0f );
        EXPECT_FLOAT_EQ( simd::get< 7>(c), 7.0f );
        EXPECT_FLOAT_EQ( simd::get< 8>(c), 7.0f );
        EXPECT_FLOAT_EQ( simd::get< 9>(c), 6.0f );
        EXPECT_FLOAT_EQ( simd::get<10>(c), 5.0f );
        EXPECT_FLOAT_EQ( simd::get<11>(c), 4.0f );
        EXPECT_FLOAT_EQ( simd::get<12>(c), 3.0f );
        EXPECT_FLOAT_EQ( simd::get<13>(c), 2.0f );
        EXPECT_FLOAT_EQ( simd::get<14>(c), 1.0f );
        EXPECT_FLOAT_EQ( simd::get<15>(c), 0.0f );
    }
}


//-------------------------------------------------------------------------------------------------
// SIMD int
//

TEST(SIMD, SelectInt)
{
    // int4

    {
        simd::mask4 m(1,1,0,0);
        simd::int4  a(0, 1, 2, 3);
        simd::int4  b(3, 2, 1, 0);

        simd::int4 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::int4(0, 1, 1, 0)) );
    }


    // int8

    {
        simd::mask8 m(1,1,1,1, 0,0,0,0);
        simd::int8  a(0, 1, 2, 3, 4, 5, 6, 7);
        simd::int8  b(7, 6, 5, 4, 3, 2, 1, 0);

        simd::int8 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::int8(0, 1, 2, 3, 3, 2, 1, 0)) );
    }


    // int16

    {
        simd::mask16 m(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
        simd::int16  a(
                 0,  1,  2,  3,
                 4,  5,  6,  7,
                 8,  9, 10, 11,
                12, 13, 14, 15
                );
        simd::int16  b(
                15, 14, 13, 12,
                11, 10,  9,  8,
                 7,  6,  5,  4,
                 3,  2,  1,  0
                );

        simd::int16 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::int16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0)) );
    }
}


//-------------------------------------------------------------------------------------------------
// SIMD mask
//

TEST(SIMD, SelectMask)
{
    // mask4

    {
        simd::mask4 m(1,1,0,0);
        simd::mask4 a(0,1,0,1);
        simd::mask4 b(1,0,1,0);

        simd::mask4 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::mask4(0,1,1,0)) );
    }


    // mask8

    {
        simd::mask8 m(1,1,1,1, 0,0,0,0);
        simd::mask8 a(0,1,0,1, 0,1,0,1);
        simd::mask8 b(1,0,1,0, 1,0,1,0);

        simd::mask8 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::mask8(0,1,0,1, 1,0,1,0)) );
    }


    // mask16

    {
        simd::mask16 m(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
        simd::mask16 a(0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1);
        simd::mask16 b(1,0,0,0,1,0,1,0, 1,0,1,0,1,0,1,0);

        simd::mask16 c = select(m, a, b);
        EXPECT_TRUE( all(c == simd::mask16(0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0)) );
    }
}
