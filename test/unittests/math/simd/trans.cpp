// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(SIMD, Trans)
{
    // TODO: valid range for aXXX functions

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

    VSNRAY_ALIGN(64) float arr[] = {
            0.0f, M_PI/8, M_PI/4, M_PI/2,
            1.0f, M_PI*2, M_PI*3, M_PI*4,
            -0.0f, -M_PI/8, -M_PI/4, -M_PI/2,
            -FLT_MAX, +FLT_MAX, -FLT_EPSILON, +FLT_EPSILON
            };

    // float4 ---------------------------------------------

    simd::float4 f4(arr);

    auto cos4 = cos(f4);
    auto sin4 = sin(f4);
    auto tan4 = tan(f4);

//  auto acos4 = acos(f4);
//  auto asin4 = asin(f4);
//  auto atan4 = atan(f4);

    EXPECT_FLOAT_EQ( simd::get<0>(cos4), cosf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(cos4), cosf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(cos4), cosf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(cos4), cosf(arr[3]) );

    EXPECT_FLOAT_EQ( simd::get<0>(sin4), sinf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(sin4), sinf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(sin4), sinf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(sin4), sinf(arr[3]) );

    EXPECT_FLOAT_EQ( simd::get<0>(tan4), tanf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(tan4), tanf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(tan4), tanf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(tan4), tanf(arr[3]) );

//  EXPECT_FLOAT_EQ( simd::get<0>(acos4), acosf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(acos4), acosf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(acos4), acosf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(acos4), acosf(arr[3]) );
//
//  EXPECT_FLOAT_EQ( simd::get<0>(asin4), asinf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(asin4), asinf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(asin4), asinf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(asin4), asinf(arr[3]) );
//
//  EXPECT_FLOAT_EQ( simd::get<0>(atan4), atanf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(atan4), atanf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(atan4), atanf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(atan4), atanf(arr[3]) );


    // float8 ---------------------------------------------

    simd::float8 f8(arr);

    auto cos8 = cos(f8);
    auto sin8 = sin(f8);
    auto tan8 = tan(f8);

//  auto acos8 = acos(f8);
//  auto asin8 = asin(f8);
//  auto atan8 = atan(f8);

    EXPECT_FLOAT_EQ( simd::get<0>(cos8), cosf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(cos8), cosf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(cos8), cosf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(cos8), cosf(arr[3]) );
    EXPECT_FLOAT_EQ( simd::get<4>(cos8), cosf(arr[4]) );
    EXPECT_FLOAT_EQ( simd::get<5>(cos8), cosf(arr[5]) );
    EXPECT_FLOAT_EQ( simd::get<6>(cos8), cosf(arr[6]) );
    EXPECT_FLOAT_EQ( simd::get<7>(cos8), cosf(arr[7]) );

    EXPECT_FLOAT_EQ( simd::get<0>(sin8), sinf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(sin8), sinf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(sin8), sinf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(sin8), sinf(arr[3]) );
    EXPECT_FLOAT_EQ( simd::get<4>(sin8), sinf(arr[4]) );
    EXPECT_FLOAT_EQ( simd::get<5>(sin8), sinf(arr[5]) );
    EXPECT_FLOAT_EQ( simd::get<6>(sin8), sinf(arr[6]) );
    EXPECT_FLOAT_EQ( simd::get<7>(sin8), sinf(arr[7]) );

    EXPECT_FLOAT_EQ( simd::get<0>(tan8), tanf(arr[0]) );
    EXPECT_FLOAT_EQ( simd::get<1>(tan8), tanf(arr[1]) );
    EXPECT_FLOAT_EQ( simd::get<2>(tan8), tanf(arr[2]) );
    EXPECT_FLOAT_EQ( simd::get<3>(tan8), tanf(arr[3]) );
    EXPECT_FLOAT_EQ( simd::get<4>(tan8), tanf(arr[4]) );
    EXPECT_FLOAT_EQ( simd::get<5>(tan8), tanf(arr[5]) );
    EXPECT_FLOAT_EQ( simd::get<6>(tan8), tanf(arr[6]) );
    EXPECT_FLOAT_EQ( simd::get<7>(tan8), tanf(arr[7]) );

//  EXPECT_FLOAT_EQ( simd::get<0>(acos8), acosf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(acos8), acosf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(acos8), acosf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(acos8), acosf(arr[3]) );
//  EXPECT_FLOAT_EQ( simd::get<4>(acos8), acosf(arr[4]) );
//  EXPECT_FLOAT_EQ( simd::get<5>(acos8), acosf(arr[5]) );
//  EXPECT_FLOAT_EQ( simd::get<6>(acos8), acosf(arr[6]) );
//  EXPECT_FLOAT_EQ( simd::get<7>(acos8), acosf(arr[7]) );
//
//  EXPECT_FLOAT_EQ( simd::get<0>(asin8), asinf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(asin8), asinf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(asin8), asinf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(asin8), asinf(arr[3]) );
//  EXPECT_FLOAT_EQ( simd::get<4>(asin8), asinf(arr[4]) );
//  EXPECT_FLOAT_EQ( simd::get<5>(asin8), asinf(arr[5]) );
//  EXPECT_FLOAT_EQ( simd::get<6>(asin8), asinf(arr[6]) );
//  EXPECT_FLOAT_EQ( simd::get<7>(asin8), asinf(arr[7]) );
//
//  EXPECT_FLOAT_EQ( simd::get<0>(atan8), atanf(arr[0]) );
//  EXPECT_FLOAT_EQ( simd::get<1>(atan8), atanf(arr[1]) );
//  EXPECT_FLOAT_EQ( simd::get<2>(atan8), atanf(arr[2]) );
//  EXPECT_FLOAT_EQ( simd::get<3>(atan8), atanf(arr[3]) );
//  EXPECT_FLOAT_EQ( simd::get<4>(atan8), atanf(arr[4]) );
//  EXPECT_FLOAT_EQ( simd::get<5>(atan8), atanf(arr[5]) );
//  EXPECT_FLOAT_EQ( simd::get<6>(atan8), atanf(arr[6]) );
//  EXPECT_FLOAT_EQ( simd::get<7>(atan8), atanf(arr[7]) );


    // float16 --------------------------------------------

    simd::float16 f16(arr);

    auto cos16 = cos(f16);
    auto sin16 = sin(f16);
    auto tan16 = tan(f16);

//  auto acos16 = acos(f16);
//  auto asin16 = asin(f16);
//  auto atan16 = atan(f16);

    EXPECT_FLOAT_EQ( simd::get< 0>(cos16), cosf(arr[ 0]) );
    EXPECT_FLOAT_EQ( simd::get< 1>(cos16), cosf(arr[ 1]) );
    EXPECT_FLOAT_EQ( simd::get< 2>(cos16), cosf(arr[ 2]) );
    EXPECT_FLOAT_EQ( simd::get< 3>(cos16), cosf(arr[ 3]) );
    EXPECT_FLOAT_EQ( simd::get< 4>(cos16), cosf(arr[ 4]) );
    EXPECT_FLOAT_EQ( simd::get< 5>(cos16), cosf(arr[ 5]) );
    EXPECT_FLOAT_EQ( simd::get< 6>(cos16), cosf(arr[ 6]) );
    EXPECT_FLOAT_EQ( simd::get< 7>(cos16), cosf(arr[ 7]) );
    EXPECT_FLOAT_EQ( simd::get< 8>(cos16), cosf(arr[ 8]) );
    EXPECT_FLOAT_EQ( simd::get< 9>(cos16), cosf(arr[ 9]) );
    EXPECT_FLOAT_EQ( simd::get<10>(cos16), cosf(arr[10]) );
    EXPECT_FLOAT_EQ( simd::get<11>(cos16), cosf(arr[11]) );
    EXPECT_FLOAT_EQ( simd::get<12>(cos16), cosf(arr[12]) );
    EXPECT_FLOAT_EQ( simd::get<13>(cos16), cosf(arr[13]) );
    EXPECT_FLOAT_EQ( simd::get<14>(cos16), cosf(arr[14]) );
    EXPECT_FLOAT_EQ( simd::get<15>(cos16), cosf(arr[15]) );

    EXPECT_FLOAT_EQ( simd::get< 0>(sin16), sinf(arr[ 0]) );
    EXPECT_FLOAT_EQ( simd::get< 1>(sin16), sinf(arr[ 1]) );
    EXPECT_FLOAT_EQ( simd::get< 2>(sin16), sinf(arr[ 2]) );
    EXPECT_FLOAT_EQ( simd::get< 3>(sin16), sinf(arr[ 3]) );
    EXPECT_FLOAT_EQ( simd::get< 4>(sin16), sinf(arr[ 4]) );
    EXPECT_FLOAT_EQ( simd::get< 5>(sin16), sinf(arr[ 5]) );
    EXPECT_FLOAT_EQ( simd::get< 6>(sin16), sinf(arr[ 6]) );
    EXPECT_FLOAT_EQ( simd::get< 7>(sin16), sinf(arr[ 7]) );
    EXPECT_FLOAT_EQ( simd::get< 8>(sin16), sinf(arr[ 8]) );
    EXPECT_FLOAT_EQ( simd::get< 9>(sin16), sinf(arr[ 9]) );
    EXPECT_FLOAT_EQ( simd::get<10>(sin16), sinf(arr[10]) );
    EXPECT_FLOAT_EQ( simd::get<11>(sin16), sinf(arr[11]) );
    EXPECT_FLOAT_EQ( simd::get<12>(sin16), sinf(arr[12]) );
    EXPECT_FLOAT_EQ( simd::get<13>(sin16), sinf(arr[13]) );
    EXPECT_FLOAT_EQ( simd::get<14>(sin16), sinf(arr[14]) );
    EXPECT_FLOAT_EQ( simd::get<15>(sin16), sinf(arr[15]) );

    EXPECT_FLOAT_EQ( simd::get< 0>(tan16), tanf(arr[ 0]) );
    EXPECT_FLOAT_EQ( simd::get< 1>(tan16), tanf(arr[ 1]) );
    EXPECT_FLOAT_EQ( simd::get< 2>(tan16), tanf(arr[ 2]) );
    EXPECT_FLOAT_EQ( simd::get< 3>(tan16), tanf(arr[ 3]) );
    EXPECT_FLOAT_EQ( simd::get< 4>(tan16), tanf(arr[ 4]) );
    EXPECT_FLOAT_EQ( simd::get< 5>(tan16), tanf(arr[ 5]) );
    EXPECT_FLOAT_EQ( simd::get< 6>(tan16), tanf(arr[ 6]) );
    EXPECT_FLOAT_EQ( simd::get< 7>(tan16), tanf(arr[ 7]) );
    EXPECT_FLOAT_EQ( simd::get< 8>(tan16), tanf(arr[ 8]) );
    EXPECT_FLOAT_EQ( simd::get< 9>(tan16), tanf(arr[ 9]) );
    EXPECT_FLOAT_EQ( simd::get<10>(tan16), tanf(arr[10]) );
    EXPECT_FLOAT_EQ( simd::get<11>(tan16), tanf(arr[11]) );
    EXPECT_FLOAT_EQ( simd::get<12>(tan16), tanf(arr[12]) );
    EXPECT_FLOAT_EQ( simd::get<13>(tan16), tanf(arr[13]) );
    EXPECT_FLOAT_EQ( simd::get<14>(tan16), tanf(arr[14]) );
    EXPECT_FLOAT_EQ( simd::get<15>(tan16), tanf(arr[15]) );

//  EXPECT_FLOAT_EQ( simd::get< 0>(acos16), acosf(arr[ 0]) );
//  EXPECT_FLOAT_EQ( simd::get< 1>(acos16), acosf(arr[ 1]) );
//  EXPECT_FLOAT_EQ( simd::get< 2>(acos16), acosf(arr[ 2]) );
//  EXPECT_FLOAT_EQ( simd::get< 3>(acos16), acosf(arr[ 3]) );
//  EXPECT_FLOAT_EQ( simd::get< 4>(acos16), acosf(arr[ 4]) );
//  EXPECT_FLOAT_EQ( simd::get< 5>(acos16), acosf(arr[ 5]) );
//  EXPECT_FLOAT_EQ( simd::get< 6>(acos16), acosf(arr[ 6]) );
//  EXPECT_FLOAT_EQ( simd::get< 7>(acos16), acosf(arr[ 7]) );
//  EXPECT_FLOAT_EQ( simd::get< 8>(acos16), acosf(arr[ 8]) );
//  EXPECT_FLOAT_EQ( simd::get< 9>(acos16), acosf(arr[ 9]) );
//  EXPECT_FLOAT_EQ( simd::get<10>(acos16), acosf(arr[10]) );
//  EXPECT_FLOAT_EQ( simd::get<11>(acos16), acosf(arr[11]) );
//  EXPECT_FLOAT_EQ( simd::get<12>(acos16), acosf(arr[12]) );
//  EXPECT_FLOAT_EQ( simd::get<13>(acos16), acosf(arr[13]) );
//  EXPECT_FLOAT_EQ( simd::get<14>(acos16), acosf(arr[14]) );
//  EXPECT_FLOAT_EQ( simd::get<15>(acos16), acosf(arr[15]) );
//
//  EXPECT_FLOAT_EQ( simd::get< 0>(asin16), asinf(arr[ 0]) );
//  EXPECT_FLOAT_EQ( simd::get< 1>(asin16), asinf(arr[ 1]) );
//  EXPECT_FLOAT_EQ( simd::get< 2>(asin16), asinf(arr[ 2]) );
//  EXPECT_FLOAT_EQ( simd::get< 3>(asin16), asinf(arr[ 3]) );
//  EXPECT_FLOAT_EQ( simd::get< 4>(asin16), asinf(arr[ 4]) );
//  EXPECT_FLOAT_EQ( simd::get< 5>(asin16), asinf(arr[ 5]) );
//  EXPECT_FLOAT_EQ( simd::get< 6>(asin16), asinf(arr[ 6]) );
//  EXPECT_FLOAT_EQ( simd::get< 7>(asin16), asinf(arr[ 7]) );
//  EXPECT_FLOAT_EQ( simd::get< 8>(asin16), asinf(arr[ 8]) );
//  EXPECT_FLOAT_EQ( simd::get< 9>(asin16), asinf(arr[ 9]) );
//  EXPECT_FLOAT_EQ( simd::get<10>(asin16), asinf(arr[10]) );
//  EXPECT_FLOAT_EQ( simd::get<11>(asin16), asinf(arr[11]) );
//  EXPECT_FLOAT_EQ( simd::get<12>(asin16), asinf(arr[12]) );
//  EXPECT_FLOAT_EQ( simd::get<13>(asin16), asinf(arr[13]) );
//  EXPECT_FLOAT_EQ( simd::get<14>(asin16), asinf(arr[14]) );
//  EXPECT_FLOAT_EQ( simd::get<15>(asin16), asinf(arr[15]) );
//
//  EXPECT_FLOAT_EQ( simd::get< 0>(atan16), atanf(arr[ 0]) );
//  EXPECT_FLOAT_EQ( simd::get< 1>(atan16), atanf(arr[ 1]) );
//  EXPECT_FLOAT_EQ( simd::get< 2>(atan16), atanf(arr[ 2]) );
//  EXPECT_FLOAT_EQ( simd::get< 3>(atan16), atanf(arr[ 3]) );
//  EXPECT_FLOAT_EQ( simd::get< 4>(atan16), atanf(arr[ 4]) );
//  EXPECT_FLOAT_EQ( simd::get< 5>(atan16), atanf(arr[ 5]) );
//  EXPECT_FLOAT_EQ( simd::get< 6>(atan16), atanf(arr[ 6]) );
//  EXPECT_FLOAT_EQ( simd::get< 7>(atan16), atanf(arr[ 7]) );
//  EXPECT_FLOAT_EQ( simd::get< 8>(atan16), atanf(arr[ 8]) );
//  EXPECT_FLOAT_EQ( simd::get< 9>(atan16), atanf(arr[ 9]) );
//  EXPECT_FLOAT_EQ( simd::get<10>(atan16), atanf(arr[10]) );
//  EXPECT_FLOAT_EQ( simd::get<11>(atan16), atanf(arr[11]) );
//  EXPECT_FLOAT_EQ( simd::get<12>(atan16), atanf(arr[12]) );
//  EXPECT_FLOAT_EQ( simd::get<13>(atan16), atanf(arr[13]) );
//  EXPECT_FLOAT_EQ( simd::get<14>(atan16), atanf(arr[14]) );
//  EXPECT_FLOAT_EQ( simd::get<15>(atan16), atanf(arr[15]) );
}
