// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


template <typename T>
void test_representability()
{
    //-------------------------------------------------------------------------
    // isnan
    //

    EXPECT_TRUE ( all(isnan(T(NAN))) );
    EXPECT_FALSE( any(isnan(T(INFINITY))) );
    EXPECT_FALSE( any(isnan(T(0.0))) );
    EXPECT_FALSE( any(isnan(T(DBL_MIN / 2.0))) );
    EXPECT_TRUE ( all(isnan(T(0.0 / 0.0))) );
    EXPECT_TRUE ( all(isnan(T(INFINITY - INFINITY))) );


    //-------------------------------------------------------------------------
    // isinf
    //

    EXPECT_FALSE( any(isinf(T(NAN))) );
    EXPECT_TRUE ( all(isinf(T(INFINITY))) );
    EXPECT_FALSE( any(isinf(T(0.0))) );
    EXPECT_TRUE ( all(isinf(T(std::exp(800)))) );
    EXPECT_FALSE( any(isinf(T(DBL_MIN / 2.0))) );


    //-------------------------------------------------------------------------
    // isfinite
    //

    EXPECT_FALSE( any(isfinite(T(NAN))) );
    EXPECT_FALSE( any(isfinite(T(INFINITY))) );
    EXPECT_TRUE ( all(isfinite(T(0.0))) );
    EXPECT_FALSE( any(isfinite(T(exp(800)))) );
    EXPECT_TRUE ( all(isfinite(T(DBL_MIN / 2.0))) );
}


//-------------------------------------------------------------------------------------------------
// Test isnan(), isinf(), and isfinite()
//

TEST(SSE, Representability)
{
    test_representability<simd::float4>();
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    test_representability<simd::float8>();
#endif
}
