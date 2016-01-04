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

static void test_pred_4()
{
    using M = simd::mask4;

    M z(0,0,0,0);
    M a(1,1,0,0);
    M i(1,1,1,1);

    EXPECT_TRUE(!any(z) );
    EXPECT_TRUE(!all(z) );
    EXPECT_TRUE( any(a) );
    EXPECT_TRUE(!all(a) );
    EXPECT_TRUE( any(i) );
    EXPECT_TRUE( all(i) );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
static void test_pred_8()
{
    using M = simd::mask8;

    M z(0,0,0,0, 0,0,0,0);
    M a(1,1,0,0, 1,1,0,0);
    M i(1,1,1,1, 1,1,1,1);

    EXPECT_TRUE(!any(z) );
    EXPECT_TRUE(!all(z) );
    EXPECT_TRUE( any(a) );
    EXPECT_TRUE(!all(a) );
    EXPECT_TRUE( any(i) );
    EXPECT_TRUE( all(i) );
}
#endif

static void test_logical_4()
{
    using M = simd::mask4;

    M a(1,1,0,0);
    M b(1,0,1,0);
    M c(0,0,1,1);

    EXPECT_TRUE( all((a && b) == M(1,0,0,0)) );
    EXPECT_TRUE( all((a && c) == M(0,0,0,0)) );
    EXPECT_TRUE( all((a || b) == M(1,1,1,0)) );
    EXPECT_TRUE( all((a || c) == M(1,1,1,1)) );

    EXPECT_TRUE( any(a && b) );
    EXPECT_TRUE(!any(a && c) );
    EXPECT_TRUE( any(a || b) );
    EXPECT_TRUE( all(a || c) );

    EXPECT_TRUE( any(!(a && b)) );
    EXPECT_TRUE( all(!(a && c)) );
    EXPECT_TRUE( any(!(a || b)) );
    EXPECT_TRUE(!any(!(a || c)) );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
static void test_logical_8()
{
    using M = simd::mask8;

    M a(1,1,0,0, 1,1,0,0);
    M b(1,0,1,0, 1,0,1,0);
    M c(0,0,1,1, 0,0,1,1);

    EXPECT_TRUE( all((a && b) == M(1,0,0,0, 1,0,0,0)) );
    EXPECT_TRUE( all((a && c) == M(0,0,0,0, 0,0,0,0)) );
    EXPECT_TRUE( all((a || b) == M(1,1,1,0, 1,1,1,0)) );
    EXPECT_TRUE( all((a || c) == M(1,1,1,1, 1,1,1,1)) );

    EXPECT_TRUE( any(a && b) );
    EXPECT_TRUE(!any(a && c) );
    EXPECT_TRUE( any(a || b) );
    EXPECT_TRUE( all(a || c) );

    EXPECT_TRUE( any(!(a && b)) );
    EXPECT_TRUE( all(!(a && c)) );
    EXPECT_TRUE( any(!(a || b)) );
    EXPECT_TRUE(!any(!(a || c)) );
}
#endif


//-------------------------------------------------------------------------------------------------
// Test isnan(), isinf(), and isfinite()
//

TEST(SIMD, Representability)
{
    test_representability<simd::float4>();
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    test_representability<simd::float8>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test all() and any()
//

TEST(SIMD, Pred)
{
    test_pred_4();
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    test_pred_8();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test logical operations
//

TEST(SIMD, Logical)
{
    test_logical_4();
#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
    test_logical_8();
#endif
}
