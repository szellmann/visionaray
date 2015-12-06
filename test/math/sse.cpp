// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(SSE, Float4MathFuncs)
{

    //-------------------------------------------------------------------------
    // isnan
    //

    EXPECT_TRUE ( all(isnan(simd::float4(NAN))) );
    EXPECT_FALSE( any(isnan(simd::float4(INFINITY))) );
    EXPECT_FALSE( any(isnan(simd::float4(0.0))) );
    EXPECT_FALSE( any(isnan(simd::float4(DBL_MIN / 2.0))) );
    EXPECT_TRUE ( all(isnan(simd::float4(0.0 / 0.0))) );
    EXPECT_TRUE ( all(isnan(simd::float4(INFINITY - INFINITY))) );


    //-------------------------------------------------------------------------
    // isinf
    //

    EXPECT_FALSE( any(isinf(simd::float4(NAN))) );
    EXPECT_TRUE ( all(isinf(simd::float4(INFINITY))) );
    EXPECT_FALSE( any(isinf(simd::float4(0.0))) );
    EXPECT_TRUE ( all(isinf(simd::float4(std::exp(800)))) );
    EXPECT_FALSE( any(isinf(simd::float4(DBL_MIN / 2.0))) );


    //-------------------------------------------------------------------------
    // isfinite
    //

    EXPECT_FALSE( any(isfinite(simd::float4(NAN))) );
    EXPECT_FALSE( any(isfinite(simd::float4(INFINITY))) );
    EXPECT_TRUE ( all(isfinite(simd::float4(0.0))) );
    EXPECT_FALSE( any(isfinite(simd::float4(exp(800)))) );
    EXPECT_TRUE ( all(isfinite(simd::float4(DBL_MIN / 2.0))) );
}
