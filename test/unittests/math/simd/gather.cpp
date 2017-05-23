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

    VSNRAY_ALIGN(64) unorm<Bits> arr[16];

    for (int i = 0; i < 16; ++i)
    {
        arr[i] = i / 16.0f;
    }


    // test float4

    simd::int4 index4(0, 2, 4, 6);
    simd::float4 res4 = gather(arr, index4);

    EXPECT_FLOAT_EQ( simd::get<0>(res4), static_cast<float>(unorm<Bits>( 0 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<1>(res4), static_cast<float>(unorm<Bits>( 2 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<2>(res4), static_cast<float>(unorm<Bits>( 4 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<3>(res4), static_cast<float>(unorm<Bits>( 6 / 16.0f)) );

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // test float8

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    simd::float8 res8 = gather(arr, index8);

    EXPECT_FLOAT_EQ( simd::get<0>(res8), static_cast<float>(unorm<Bits>( 0 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<1>(res8), static_cast<float>(unorm<Bits>( 2 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<2>(res8), static_cast<float>(unorm<Bits>( 4 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<3>(res8), static_cast<float>(unorm<Bits>( 6 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<4>(res8), static_cast<float>(unorm<Bits>( 8 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<5>(res8), static_cast<float>(unorm<Bits>(10 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<6>(res8), static_cast<float>(unorm<Bits>(12 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<7>(res8), static_cast<float>(unorm<Bits>(14 / 16.0f)) );

#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // test float16

    simd::int16 index16(
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15
            );
    simd::float16 res16 = gather(arr, index16);

    EXPECT_FLOAT_EQ( simd::get< 0>(res16), static_cast<float>(unorm<Bits>( 0 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 1>(res16), static_cast<float>(unorm<Bits>( 1 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 2>(res16), static_cast<float>(unorm<Bits>( 2 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 3>(res16), static_cast<float>(unorm<Bits>( 3 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 4>(res16), static_cast<float>(unorm<Bits>( 4 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 5>(res16), static_cast<float>(unorm<Bits>( 5 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 6>(res16), static_cast<float>(unorm<Bits>( 6 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 7>(res16), static_cast<float>(unorm<Bits>( 7 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 8>(res16), static_cast<float>(unorm<Bits>( 8 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get< 9>(res16), static_cast<float>(unorm<Bits>( 9 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<10>(res16), static_cast<float>(unorm<Bits>(10 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<11>(res16), static_cast<float>(unorm<Bits>(11 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<12>(res16), static_cast<float>(unorm<Bits>(12 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<13>(res16), static_cast<float>(unorm<Bits>(13 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<14>(res16), static_cast<float>(unorm<Bits>(14 / 16.0f)) );
    EXPECT_FLOAT_EQ( simd::get<15>(res16), static_cast<float>(unorm<Bits>(15 / 16.0f)) );

#endif

}


template <size_t Dim, unsigned Bits>
static void test_gather_vector_unorm()
{

    // init memory

    VSNRAY_ALIGN(64) vector<Dim, unorm<Bits>> arr[16];

    for (int i = 0; i < 16; ++i)
    {
        for (size_t d = 0; d < Dim; ++d)
        {
            arr[i][d] = (i * Dim + d) / static_cast<float>(Dim * 16);
        }
    }


    // test vector<Dim, float4>

    simd::int4 index4(0, 2, 4, 6);
    vector<Dim, simd::float4> res4 = gather(arr, index4);

    for (size_t d = 0; d < Dim; ++d)
    {
        simd::float4 f = res4[d];

        unorm<Bits> u0( ( 0 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u1( ( 2 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u2( ( 4 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u3( ( 6 * Dim + d) / static_cast<float>(Dim * 16) );

        EXPECT_FLOAT_EQ( simd::get<0>(f), static_cast<float>(u0) );
        EXPECT_FLOAT_EQ( simd::get<1>(f), static_cast<float>(u1) );
        EXPECT_FLOAT_EQ( simd::get<2>(f), static_cast<float>(u2) );
        EXPECT_FLOAT_EQ( simd::get<3>(f), static_cast<float>(u3) );
    }

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // test vector<Dim, float8>

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    vector<Dim, simd::float8> res8 = gather(arr, index8);

    for (size_t d = 0; d < Dim; ++d)
    {
        simd::float8 f = res8[d];

        unorm<Bits> u0( ( 0 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u1( ( 2 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u2( ( 4 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u3( ( 6 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u4( ( 8 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u5( (10 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u6( (12 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u7( (14 * Dim + d) / static_cast<float>(Dim * 16) );

        EXPECT_FLOAT_EQ( simd::get<0>(f), static_cast<float>(u0) );
        EXPECT_FLOAT_EQ( simd::get<1>(f), static_cast<float>(u1) );
        EXPECT_FLOAT_EQ( simd::get<2>(f), static_cast<float>(u2) );
        EXPECT_FLOAT_EQ( simd::get<3>(f), static_cast<float>(u3) );
        EXPECT_FLOAT_EQ( simd::get<4>(f), static_cast<float>(u4) );
        EXPECT_FLOAT_EQ( simd::get<5>(f), static_cast<float>(u5) );
        EXPECT_FLOAT_EQ( simd::get<6>(f), static_cast<float>(u6) );
        EXPECT_FLOAT_EQ( simd::get<7>(f), static_cast<float>(u7) );
    }

#endif

#if 0 && VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // test vector<Dim, float16>

    simd::int16 index16(
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15
            );
    vector<Dim, simd::float16> res16 = gather(arr, index16);

    for (size_t d = 0; d < Dim; ++d)
    {
        simd::float16 f = res16[d];

        unorm<Bits>  u0( (  0 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u1( (  1 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u2( (  2 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u3( (  3 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u4( (  4 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u5( (  5 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u6( (  6 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u7( (  7 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u8( (  8 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits>  u9( (  9 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u10( ( 10 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u11( ( 11 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u12( ( 12 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u13( ( 13 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u14( ( 14 * Dim + d) / static_cast<float>(Dim * 16) );
        unorm<Bits> u15( ( 15 * Dim + d) / static_cast<float>(Dim * 16) );

        EXPECT_FLOAT_EQ( simd::get< 0>(f), static_cast<float>( u0) );
        EXPECT_FLOAT_EQ( simd::get< 1>(f), static_cast<float>( u1) );
        EXPECT_FLOAT_EQ( simd::get< 2>(f), static_cast<float>( u2) );
        EXPECT_FLOAT_EQ( simd::get< 3>(f), static_cast<float>( u3) );
        EXPECT_FLOAT_EQ( simd::get< 4>(f), static_cast<float>( u4) );
        EXPECT_FLOAT_EQ( simd::get< 5>(f), static_cast<float>( u5) );
        EXPECT_FLOAT_EQ( simd::get< 6>(f), static_cast<float>( u6) );
        EXPECT_FLOAT_EQ( simd::get< 7>(f), static_cast<float>( u7) );
        EXPECT_FLOAT_EQ( simd::get< 8>(f), static_cast<float>( u8) );
        EXPECT_FLOAT_EQ( simd::get< 9>(f), static_cast<float>( u9) );
        EXPECT_FLOAT_EQ( simd::get<10>(f), static_cast<float>(u10) );
        EXPECT_FLOAT_EQ( simd::get<11>(f), static_cast<float>(u11) );
        EXPECT_FLOAT_EQ( simd::get<12>(f), static_cast<float>(u12) );
        EXPECT_FLOAT_EQ( simd::get<13>(f), static_cast<float>(u13) );
        EXPECT_FLOAT_EQ( simd::get<14>(f), static_cast<float>(u14) );
        EXPECT_FLOAT_EQ( simd::get<15>(f), static_cast<float>(u15) );
    }

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

    VSNRAY_ALIGN(64) float arr[16];

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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // test float16

    simd::int16 index16(
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15
            );
    simd::float16 res16 = gather(arr, index16);

    EXPECT_FLOAT_EQ(simd::get< 0>(res16),  0.0f);
    EXPECT_FLOAT_EQ(simd::get< 1>(res16),  1.0f);
    EXPECT_FLOAT_EQ(simd::get< 2>(res16),  2.0f);
    EXPECT_FLOAT_EQ(simd::get< 3>(res16),  3.0f);
    EXPECT_FLOAT_EQ(simd::get< 4>(res16),  4.0f);
    EXPECT_FLOAT_EQ(simd::get< 5>(res16),  5.0f);
    EXPECT_FLOAT_EQ(simd::get< 6>(res16),  6.0f);
    EXPECT_FLOAT_EQ(simd::get< 7>(res16),  7.0f);
    EXPECT_FLOAT_EQ(simd::get< 8>(res16),  8.0f);
    EXPECT_FLOAT_EQ(simd::get< 9>(res16),  9.0f);
    EXPECT_FLOAT_EQ(simd::get<10>(res16), 10.0f);
    EXPECT_FLOAT_EQ(simd::get<11>(res16), 11.0f);
    EXPECT_FLOAT_EQ(simd::get<12>(res16), 12.0f);
    EXPECT_FLOAT_EQ(simd::get<13>(res16), 13.0f);
    EXPECT_FLOAT_EQ(simd::get<14>(res16), 14.0f);
    EXPECT_FLOAT_EQ(simd::get<15>(res16), 15.0f);

#endif

}


//-------------------------------------------------------------------------------------------------
// Test gather() with integers
//

TEST(SIMD, GatherInt)
{

    // init memory

    VSNRAY_ALIGN(64) int arr[32];

    for (int i = 0; i < 32; ++i)
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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // test int8

    simd::int16 index16(
             0,  2,  4,  6,
             8, 10, 12, 14,
            16, 18, 20, 22,
            24, 26, 28, 30
            );
    simd::int16 res16 = gather(arr, index16);

    EXPECT_TRUE(simd::get< 0>(res16) ==  0);
    EXPECT_TRUE(simd::get< 1>(res16) ==  2);
    EXPECT_TRUE(simd::get< 2>(res16) ==  4);
    EXPECT_TRUE(simd::get< 3>(res16) ==  6);
    EXPECT_TRUE(simd::get< 4>(res16) ==  8);
    EXPECT_TRUE(simd::get< 5>(res16) == 10);
    EXPECT_TRUE(simd::get< 6>(res16) == 12);
    EXPECT_TRUE(simd::get< 7>(res16) == 14);
    EXPECT_TRUE(simd::get< 8>(res16) == 16);
    EXPECT_TRUE(simd::get< 9>(res16) == 18);
    EXPECT_TRUE(simd::get<10>(res16) == 20);
    EXPECT_TRUE(simd::get<11>(res16) == 22);
    EXPECT_TRUE(simd::get<12>(res16) == 24);
    EXPECT_TRUE(simd::get<13>(res16) == 26);
    EXPECT_TRUE(simd::get<14>(res16) == 28);
    EXPECT_TRUE(simd::get<15>(res16) == 30);

#endif

}


//-------------------------------------------------------------------------------------------------
// Test gather() with unorm vectors
//

TEST(SIMD, GatherVecNUnorm)
{
    test_gather_vector_unorm<2,  8>();
    test_gather_vector_unorm<3,  8>();
    test_gather_vector_unorm<4,  8>();

    test_gather_vector_unorm<2, 16>();
    test_gather_vector_unorm<3, 16>();
    test_gather_vector_unorm<4, 16>();

    test_gather_vector_unorm<2, 32>();
    test_gather_vector_unorm<3, 32>();
    test_gather_vector_unorm<4, 32>();
}


//-------------------------------------------------------------------------------------------------
// Test gather() with vec4's
//

TEST(SIMD, GatherVec4)
{

    // init memory

    VSNRAY_ALIGN(64) vec4 arr[32];

    for (int i = 0; i < 32; ++i)
    {
        arr[i] = vec4(
            static_cast<float>(i * 4),
            static_cast<float>(i * 4 + 1),
            static_cast<float>(i * 4 + 2),
            static_cast<float>(i * 4 + 3)
            );

    }


    // test vector<4, float4>

    simd::int4 index4(0, 2, 4, 6);
    vector<4, simd::float4> res4 = gather(arr, index4);

    EXPECT_FLOAT_EQ(simd::get<0>(res4.x),  0.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res4.y),  1.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res4.z),  2.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res4.w),  3.0f);

    EXPECT_FLOAT_EQ(simd::get<1>(res4.x),  8.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res4.y),  9.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res4.z), 10.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res4.w), 11.0f);

    EXPECT_FLOAT_EQ(simd::get<2>(res4.x), 16.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res4.y), 17.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res4.z), 18.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res4.w), 19.0f);

    EXPECT_FLOAT_EQ(simd::get<3>(res4.x), 24.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res4.y), 25.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res4.z), 26.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res4.w), 27.0f);

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // test vector<4, float8>

    simd::int8 index8(0, 2, 4, 6, 8, 10, 12, 14);
    vector<4, simd::float8> res8 = gather(arr, index8);

    EXPECT_FLOAT_EQ(simd::get<0>(res8.x),  0.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res8.y),  1.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res8.z),  2.0f);
    EXPECT_FLOAT_EQ(simd::get<0>(res8.w),  3.0f);

    EXPECT_FLOAT_EQ(simd::get<1>(res8.x),  8.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res8.y),  9.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res8.z), 10.0f);
    EXPECT_FLOAT_EQ(simd::get<1>(res8.w), 11.0f);

    EXPECT_FLOAT_EQ(simd::get<2>(res8.x), 16.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res8.y), 17.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res8.z), 18.0f);
    EXPECT_FLOAT_EQ(simd::get<2>(res8.w), 19.0f);

    EXPECT_FLOAT_EQ(simd::get<3>(res8.x), 24.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res8.y), 25.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res8.z), 26.0f);
    EXPECT_FLOAT_EQ(simd::get<3>(res8.w), 27.0f);

    EXPECT_FLOAT_EQ(simd::get<4>(res8.x), 32.0f);
    EXPECT_FLOAT_EQ(simd::get<4>(res8.y), 33.0f);
    EXPECT_FLOAT_EQ(simd::get<4>(res8.z), 34.0f);
    EXPECT_FLOAT_EQ(simd::get<4>(res8.w), 35.0f);

    EXPECT_FLOAT_EQ(simd::get<5>(res8.x), 40.0f);
    EXPECT_FLOAT_EQ(simd::get<5>(res8.y), 41.0f);
    EXPECT_FLOAT_EQ(simd::get<5>(res8.z), 42.0f);
    EXPECT_FLOAT_EQ(simd::get<5>(res8.w), 43.0f);

    EXPECT_FLOAT_EQ(simd::get<6>(res8.x), 48.0f);
    EXPECT_FLOAT_EQ(simd::get<6>(res8.y), 49.0f);
    EXPECT_FLOAT_EQ(simd::get<6>(res8.z), 50.0f);
    EXPECT_FLOAT_EQ(simd::get<6>(res8.w), 51.0f);

    EXPECT_FLOAT_EQ(simd::get<7>(res8.x), 56.0f);
    EXPECT_FLOAT_EQ(simd::get<7>(res8.y), 57.0f);
    EXPECT_FLOAT_EQ(simd::get<7>(res8.z), 58.0f);
    EXPECT_FLOAT_EQ(simd::get<7>(res8.w), 59.0f);

#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // test vector<4, float16>

    simd::int16 index16(
             0,  2,  4,  6,
             8, 10, 12, 14,
            16, 18, 20, 22,
            24, 26, 28, 30
            );
    vector<4, simd::float16> res16 = gather(arr, index16);

    EXPECT_FLOAT_EQ(simd::get< 0>(res16.x),   0.0f);
    EXPECT_FLOAT_EQ(simd::get< 0>(res16.y),   1.0f);
    EXPECT_FLOAT_EQ(simd::get< 0>(res16.z),   2.0f);
    EXPECT_FLOAT_EQ(simd::get< 0>(res16.w),   3.0f);

    EXPECT_FLOAT_EQ(simd::get< 1>(res16.x),   8.0f);
    EXPECT_FLOAT_EQ(simd::get< 1>(res16.y),   9.0f);
    EXPECT_FLOAT_EQ(simd::get< 1>(res16.z),  10.0f);
    EXPECT_FLOAT_EQ(simd::get< 1>(res16.w),  11.0f);

    EXPECT_FLOAT_EQ(simd::get< 2>(res16.x),  16.0f);
    EXPECT_FLOAT_EQ(simd::get< 2>(res16.y),  17.0f);
    EXPECT_FLOAT_EQ(simd::get< 2>(res16.z),  18.0f);
    EXPECT_FLOAT_EQ(simd::get< 2>(res16.w),  19.0f);

    EXPECT_FLOAT_EQ(simd::get< 3>(res16.x),  24.0f);
    EXPECT_FLOAT_EQ(simd::get< 3>(res16.y),  25.0f);
    EXPECT_FLOAT_EQ(simd::get< 3>(res16.z),  26.0f);
    EXPECT_FLOAT_EQ(simd::get< 3>(res16.w),  27.0f);

    EXPECT_FLOAT_EQ(simd::get< 4>(res16.x),  32.0f);
    EXPECT_FLOAT_EQ(simd::get< 4>(res16.y),  33.0f);
    EXPECT_FLOAT_EQ(simd::get< 4>(res16.z),  34.0f);
    EXPECT_FLOAT_EQ(simd::get< 4>(res16.w),  35.0f);

    EXPECT_FLOAT_EQ(simd::get< 5>(res16.x),  40.0f);
    EXPECT_FLOAT_EQ(simd::get< 5>(res16.y),  41.0f);
    EXPECT_FLOAT_EQ(simd::get< 5>(res16.z),  42.0f);
    EXPECT_FLOAT_EQ(simd::get< 5>(res16.w),  43.0f);

    EXPECT_FLOAT_EQ(simd::get< 6>(res16.x),  48.0f);
    EXPECT_FLOAT_EQ(simd::get< 6>(res16.y),  49.0f);
    EXPECT_FLOAT_EQ(simd::get< 6>(res16.z),  50.0f);
    EXPECT_FLOAT_EQ(simd::get< 6>(res16.w),  51.0f);

    EXPECT_FLOAT_EQ(simd::get< 7>(res16.x),  56.0f);
    EXPECT_FLOAT_EQ(simd::get< 7>(res16.y),  57.0f);
    EXPECT_FLOAT_EQ(simd::get< 7>(res16.z),  58.0f);
    EXPECT_FLOAT_EQ(simd::get< 7>(res16.w),  59.0f);

    EXPECT_FLOAT_EQ(simd::get< 8>(res16.x),  64.0f);
    EXPECT_FLOAT_EQ(simd::get< 8>(res16.y),  65.0f);
    EXPECT_FLOAT_EQ(simd::get< 8>(res16.z),  66.0f);
    EXPECT_FLOAT_EQ(simd::get< 8>(res16.w),  67.0f);

    EXPECT_FLOAT_EQ(simd::get< 9>(res16.x),  72.0f);
    EXPECT_FLOAT_EQ(simd::get< 9>(res16.y),  73.0f);
    EXPECT_FLOAT_EQ(simd::get< 9>(res16.z),  74.0f);
    EXPECT_FLOAT_EQ(simd::get< 9>(res16.w),  75.0f);

    EXPECT_FLOAT_EQ(simd::get<10>(res16.x),  80.0f);
    EXPECT_FLOAT_EQ(simd::get<10>(res16.y),  81.0f);
    EXPECT_FLOAT_EQ(simd::get<10>(res16.z),  82.0f);
    EXPECT_FLOAT_EQ(simd::get<10>(res16.w),  83.0f);

    EXPECT_FLOAT_EQ(simd::get<11>(res16.x),  88.0f);
    EXPECT_FLOAT_EQ(simd::get<11>(res16.y),  89.0f);
    EXPECT_FLOAT_EQ(simd::get<11>(res16.z),  90.0f);
    EXPECT_FLOAT_EQ(simd::get<11>(res16.w),  91.0f);

    EXPECT_FLOAT_EQ(simd::get<12>(res16.x),  96.0f);
    EXPECT_FLOAT_EQ(simd::get<12>(res16.y),  97.0f);
    EXPECT_FLOAT_EQ(simd::get<12>(res16.z),  98.0f);
    EXPECT_FLOAT_EQ(simd::get<12>(res16.w),  99.0f);

    EXPECT_FLOAT_EQ(simd::get<13>(res16.x), 104.0f);
    EXPECT_FLOAT_EQ(simd::get<13>(res16.y), 105.0f);
    EXPECT_FLOAT_EQ(simd::get<13>(res16.z), 106.0f);
    EXPECT_FLOAT_EQ(simd::get<13>(res16.w), 107.0f);

    EXPECT_FLOAT_EQ(simd::get<14>(res16.x), 112.0f);
    EXPECT_FLOAT_EQ(simd::get<14>(res16.y), 113.0f);
    EXPECT_FLOAT_EQ(simd::get<14>(res16.z), 114.0f);
    EXPECT_FLOAT_EQ(simd::get<14>(res16.w), 115.0f);

    EXPECT_FLOAT_EQ(simd::get<15>(res16.x), 120.0f);
    EXPECT_FLOAT_EQ(simd::get<15>(res16.y), 121.0f);
    EXPECT_FLOAT_EQ(simd::get<15>(res16.z), 122.0f);
    EXPECT_FLOAT_EQ(simd::get<15>(res16.w), 123.0f);

#endif

}
