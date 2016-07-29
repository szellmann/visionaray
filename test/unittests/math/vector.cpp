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
}


TEST(Vector, MinMaxElement)
{
    // FPU

    vector<3, float> vf(1.0f, 2.0f, 3.0f);
    auto minef = min_element(vf);
    auto maxef = max_element(vf);
    EXPECT_FLOAT_EQ(minef, 1.0f);
    EXPECT_FLOAT_EQ(maxef, 3.0f);


    // SSE

    vector<3, simd::float4> v4(1.0f, 2.0f, 3.0f);
    auto mine4 = min_element(v4);
    auto maxe4 = max_element(v4);
    EXPECT_TRUE( all(mine4 == simd::float4(1.0f)) );
    EXPECT_TRUE( all(maxe4 == simd::float4(3.0f)) );


#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

    // AVX

    vector<3, simd::float8> v8(1.0f, 2.0f, 3.0f);
    auto mine8 = min_element(v8);
    auto maxe8 = max_element(v8);
    EXPECT_TRUE( all(mine8 == simd::float8(1.0f)) );
    EXPECT_TRUE( all(maxe8 == simd::float8(3.0f)) );

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
}
