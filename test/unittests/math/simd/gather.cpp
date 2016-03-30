// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test routines for simd::gather()
//
// Those tests should also be performed in AVX compilation mode (not AVX2)!
// simd::gather() can make use of dedicated AVX2 intrinsics in certain cases.
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Template to test gather with unorm<N> array
//

template <unsigned Bits>
static void test_gather_unorm()
{

    // init memory

    VSNRAY_ALIGN(32) unorm<Bits> arr[16];

    for (int i = 0; i < 16; ++i)
    {
        arr[i] = i;
    }


    // test float4

    simd::int4 index4(0, 2, 4, 6);
    simd::float4 res4 = gather(arr, index4);

    EXPECT_TRUE( simd::get<0>(res4) == static_cast<float>(unorm<Bits>( 0)) );
    EXPECT_TRUE( simd::get<1>(res4) == static_cast<float>(unorm<Bits>( 2)) );
    EXPECT_TRUE( simd::get<2>(res4) == static_cast<float>(unorm<Bits>( 4)) );
    EXPECT_TRUE( simd::get<3>(res4) == static_cast<float>(unorm<Bits>( 6)) );

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

    // test float8

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    simd::float8 res8 = gather(arr, index8);

    EXPECT_TRUE( simd::get<0>(res8) == static_cast<float>(unorm<Bits>( 0)) );
    EXPECT_TRUE( simd::get<1>(res8) == static_cast<float>(unorm<Bits>( 2)) );
    EXPECT_TRUE( simd::get<2>(res8) == static_cast<float>(unorm<Bits>( 4)) );
    EXPECT_TRUE( simd::get<3>(res8) == static_cast<float>(unorm<Bits>( 6)) );
    EXPECT_TRUE( simd::get<4>(res8) == static_cast<float>(unorm<Bits>( 8)) );
    EXPECT_TRUE( simd::get<5>(res8) == static_cast<float>(unorm<Bits>(10)) );
    EXPECT_TRUE( simd::get<6>(res8) == static_cast<float>(unorm<Bits>(12)) );
    EXPECT_TRUE( simd::get<7>(res8) == static_cast<float>(unorm<Bits>(14)) );

#endif

}


//-------------------------------------------------------------------------------------------------
// Test gather() with 8-bit, 16-bit, and 32-bit unorms
//

TEST(SIMD, GatherUnorm)
{
    test_gather_unorm< 8>();
    test_gather_unorm<16>();
    test_gather_unorm<32>();
}


//-------------------------------------------------------------------------------------------------
// Test gather() with floats
//

TEST(SIMD, GatherFloat)
{

    // init memory

    VSNRAY_ALIGN(32) float arr[16];

    for (int i = 0; i < 16; ++i)
    {
        arr[i] = static_cast<float>(i);
    }


    // test float4

    simd::int4 index4(0, 2, 4, 6);
    simd::float4 res4 = gather(arr, index4);

    EXPECT_FLOAT_EQ(simd::get<0>(res4),  0.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res4),  2.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res4),  4.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res4),  6.0f);

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

    // test float8

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    simd::float8 res8 = gather(arr, index8);

    EXPECT_FLOAT_EQ(simd::get<0>(res8),  0.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res8),  2.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res8),  4.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res8),  6.0f);
    EXPECT_FLOAT_EQ(simd::get<4>(res8),  8.0f);
    EXPECT_FLOAT_EQ(simd::get<5>(res8), 10.0f);
    EXPECT_FLOAT_EQ(simd::get<6>(res8), 12.0f);
    EXPECT_FLOAT_EQ(simd::get<7>(res8), 14.0f);

#endif

}


//-------------------------------------------------------------------------------------------------
// Test gather() with integers
//

TEST(SIMD, GatherInt)
{

    // init memory

    VSNRAY_ALIGN(32) int arr[16];

    for (int i = 0; i < 16; ++i)
    {
        arr[i] = i;
    }


    // test int4

    simd::int4 index4(0, 2, 4, 6);
    simd::int4 res4 = gather(arr, index4);

    EXPECT_TRUE(simd::get<0>(res4) ==  0);
    EXPECT_TRUE(simd::get<1>(res4) ==  2);
    EXPECT_TRUE(simd::get<2>(res4) ==  4);
    EXPECT_TRUE(simd::get<3>(res4) ==  6);

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

    // test int8

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    simd::int8 res8 = gather(arr, index8);

    EXPECT_TRUE(simd::get<0>(res8) ==  0);
    EXPECT_TRUE(simd::get<1>(res8) ==  2);
    EXPECT_TRUE(simd::get<2>(res8) ==  4);
    EXPECT_TRUE(simd::get<3>(res8) ==  6);
    EXPECT_TRUE(simd::get<4>(res8) ==  8);
    EXPECT_TRUE(simd::get<5>(res8) == 10);
    EXPECT_TRUE(simd::get<6>(res8) == 12);
    EXPECT_TRUE(simd::get<7>(res8) == 14);

#endif

}
