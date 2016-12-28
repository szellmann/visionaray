// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


TEST(Vector, Ctor)
{
    // Test constructability from vectors with different
    // sizes to define a higher-dimensional vector

    vector<3, float> v3(1.0f, 2.0f, 3.0f);
    vector<2, float> v2(4.0f, 5.0f);
    vector<5, float> v5(v3, v2);
    EXPECT_FLOAT_EQ(v5[0], 1.0f);
    EXPECT_FLOAT_EQ(v5[1], 2.0f);
    EXPECT_FLOAT_EQ(v5[2], 3.0f);
    EXPECT_FLOAT_EQ(v5[3], 4.0f);
    EXPECT_FLOAT_EQ(v5[4], 5.0f);

    // Test vec4 specialization of this ctor

    vector<4, float> v4(v2, vector<2, float>(6.0f, 7.0f));
    EXPECT_FLOAT_EQ(v4[0], 4.0f);
    EXPECT_FLOAT_EQ(v4[1], 5.0f);
    EXPECT_FLOAT_EQ(v4[2], 6.0f);
    EXPECT_FLOAT_EQ(v4[3], 7.0f);
}


TEST(Vector, Dot)
{
    // Test dot product for different vector sizes
    // (implementation may in general depend on size)

    vector<2, float> v2_1(1.0f, 2.0f);
    vector<2, float> v2_2(2.0f, 1.0f);
    EXPECT_FLOAT_EQ(dot(v2_1, v2_2), 4.0f);

    vector<3, float> v3_1(1.0f, 2.0f, 3.0f);
    vector<3, float> v3_2(3.0f, 2.0f, 1.0f);
    EXPECT_FLOAT_EQ(dot(v3_1, v3_2), 10.0f);

    vector<4, float> v4_1(1.0f, 2.0f, 3.0f, 4.0f);
    vector<4, float> v4_2(4.0f, 3.0f, 2.0f, 1.0f);
    EXPECT_FLOAT_EQ(dot(v4_1, v4_2), 20.0f);

    float f5_1[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    float f5_2[5] = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
    vector<5, float> v5_1(f5_1);
    vector<5, float> v5_2(f5_2);
    EXPECT_FLOAT_EQ(dot(v5_1, v5_2), 35.0f);

    float f6_1[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float f6_2[6] = { 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
    vector<6, float> v6_1(f6_1);
    vector<6, float> v6_2(f6_2);
    EXPECT_FLOAT_EQ(dot(v6_1, v6_2), 56.0f);
}


TEST(Vector, MinMaxIndex)
{
    float f[7] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    vector<7, float> vf(f);
    auto mini    = min_index(vf);
    auto maxi    = max_index(vf);
    auto minmaxi = minmax_index(vf);
    EXPECT_EQ(mini, 0U);
    EXPECT_EQ(maxi, 6U);
    EXPECT_EQ(minmaxi.x, mini);
    EXPECT_EQ(minmaxi.y, maxi);
}


TEST(Vector, MinMaxElement)
{
    // FPU

    vector<3, float> vf(1.0f, 2.0f, 3.0f);
    auto minef    = min_element(vf);
    auto maxef    = max_element(vf);
    auto minmaxef = minmax_element(vf);
    EXPECT_FLOAT_EQ(minef, 1.0f);
    EXPECT_FLOAT_EQ(maxef, 3.0f);
    EXPECT_FLOAT_EQ(minmaxef.x, minef);
    EXPECT_FLOAT_EQ(minmaxef.y, maxef);


    // SSE

    vector<3, simd::float4> v4(1.0f, 2.0f, 3.0f);
    auto mine4    = min_element(v4);
    auto maxe4    = max_element(v4);
    auto minmaxe4 = minmax_element(v4);
    EXPECT_TRUE( all(mine4 == simd::float4(1.0f)) );
    EXPECT_TRUE( all(maxe4 == simd::float4(3.0f)) );
    EXPECT_TRUE( all(minmaxe4.x == mine4) );
    EXPECT_TRUE( all(minmaxe4.y == maxe4) );


#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // AVX

    vector<3, simd::float8> v8(1.0f, 2.0f, 3.0f);
    auto mine8    = min_element(v8);
    auto maxe8    = max_element(v8);
    auto minmaxe8 = minmax_element(v8);
    EXPECT_TRUE( all(mine8 == simd::float8(1.0f)) );
    EXPECT_TRUE( all(maxe8 == simd::float8(3.0f)) );
    EXPECT_TRUE( all(minmaxe8.x == mine8) );
    EXPECT_TRUE( all(minmaxe8.y == maxe8) );

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
}
