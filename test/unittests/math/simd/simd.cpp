// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>
#include <limits>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Fill simd vectors with ascending integer values
// Test write access with simd::get()
//

template <typename T>
static void iota4(T& t)
{
    simd::get<0>(t) = 0;
    simd::get<1>(t) = 1;
    simd::get<2>(t) = 2;
    simd::get<3>(t) = 3;
}

template <typename T>
static void iota8(T& t)
{
    simd::get<0>(t) = 0;
    simd::get<1>(t) = 1;
    simd::get<2>(t) = 2;
    simd::get<3>(t) = 3;
    simd::get<4>(t) = 4;
    simd::get<5>(t) = 5;
    simd::get<6>(t) = 6;
    simd::get<7>(t) = 7;
}


//-------------------------------------------------------------------------------------------------
// Test helper functions
//

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
#if !VSNRAY_CXX_MSVC
    EXPECT_TRUE ( all(isnan(T(0.0 / 0.0))) );
#endif
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


#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
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

static void test_cmp_4()
{
    using F = simd::float4;
    using I = simd::int4;
    using M = simd::mask4;

    // float

    {
        F a(1.0f, 2.0f, 3.0f, 4.0f);
        F b(5.0f, 6.0f, 7.0f, 8.0f);
        F c(1.0f, 0.0f, 3.0f, 0.0f);
        F x(std::numeric_limits<float>::max());
        F y(std::numeric_limits<float>::min());
        F z(std::numeric_limits<float>::lowest());

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );
        EXPECT_TRUE( all(a <  b) );
        EXPECT_TRUE( all(a <= b) );
        EXPECT_TRUE( all(b >  a) );
        EXPECT_TRUE( all(b >= a) );
        EXPECT_TRUE( all(c <= a) );
        EXPECT_TRUE( all(a >= c) );

        EXPECT_TRUE( all((a > c) == M(0,1,0,1)) );
        EXPECT_TRUE( all((c < a) == M(0,1,0,1)) );

        EXPECT_TRUE( all(x >  F(0.0f)) );
        EXPECT_TRUE( all(y >  F(0.0f)) );
        EXPECT_TRUE( all(z <  F(0.0f)) );
        EXPECT_TRUE( all(x >= F(0.0f)) );
        EXPECT_TRUE( all(y >= F(0.0f)) );
        EXPECT_TRUE( all(z <= F(0.0f)) );


        EXPECT_TRUE( all(y  < x) );
        EXPECT_TRUE( all(z  < y) );
        EXPECT_TRUE( all(z  < x) );
        EXPECT_TRUE( all(y <= x) );
        EXPECT_TRUE( all(z <= y) );
        EXPECT_TRUE( all(z <= x) );

    }


    // int

    {
        I a(1, 2, 3, 4);
        I b(5, 6, 7, 8);
        I c(1, 0, 3, 0);
        I x(std::numeric_limits<int>::max());
        I y(std::numeric_limits<int>::min());

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );
        EXPECT_TRUE( all(a <  b) );
        EXPECT_TRUE( all(a <= b) );
        EXPECT_TRUE( all(b >  a) );
        EXPECT_TRUE( all(b >= a) );
        EXPECT_TRUE( all(c <= a) );
        EXPECT_TRUE( all(a >= c) );

        EXPECT_TRUE( all((a > c) == M(0,1,0,1)) );
        EXPECT_TRUE( all((c < a) == M(0,1,0,1)) );

        EXPECT_TRUE( all(x >  I(0)) );
        EXPECT_TRUE( all(y <  I(0)) );
        EXPECT_TRUE( all(x >= I(0)) );
        EXPECT_TRUE( all(y <= I(0)) );

        EXPECT_TRUE( all(y <  x) );
        EXPECT_TRUE( all(y <= x) );
    }


    // mask

    {
        M a(0,0,0,0);
        M b(1,1,1,1);
        M c(1,0,1,0);

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );

        EXPECT_TRUE( all((a == c) == M(0,1,0,1)) );
        EXPECT_TRUE( all((a != c) == M(1,0,1,0)) );
    }
}

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
static void test_cmp_8()
{
    using F = simd::float8;
    using I = simd::int8;
    using M = simd::mask8;

    // float

    {
        F a( 1.0f,  2.0f,  3.0f,  4.0f,   5.0f,  6.0f,  7.0f,  8.0f);
        F b( 9.0f, 10.0f, 11.0f, 12.0f,  13.0f, 14.0f, 15.0f, 16.0f);
        F c( 1.0f,  0.0f,  3.0f,  0.0f,   5.0f,  0.0f,  7.0f,  0.0f);
        F x(std::numeric_limits<float>::max());
        F y(std::numeric_limits<float>::min());
        F z(std::numeric_limits<float>::lowest());

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );
        EXPECT_TRUE( all(a <  b) );
        EXPECT_TRUE( all(a <= b) );
        EXPECT_TRUE( all(b >  a) );
        EXPECT_TRUE( all(b >= a) );
        EXPECT_TRUE( all(c <= a) );
        EXPECT_TRUE( all(a >= c) );

        EXPECT_TRUE( all((a > c) == M(0,1,0,1, 0,1,0,1)) );
        EXPECT_TRUE( all((c < a) == M(0,1,0,1, 0,1,0,1)) );

        EXPECT_TRUE( all(x >  F(0.0f)) );
        EXPECT_TRUE( all(y >  F(0.0f)) );
        EXPECT_TRUE( all(z <  F(0.0f)) );
        EXPECT_TRUE( all(x >= F(0.0f)) );
        EXPECT_TRUE( all(y >= F(0.0f)) );
        EXPECT_TRUE( all(z <= F(0.0f)) );

        EXPECT_TRUE( all(y  < x) );
        EXPECT_TRUE( all(z  < y) );
        EXPECT_TRUE( all(z  < x) );
        EXPECT_TRUE( all(y <= x) );
        EXPECT_TRUE( all(z <= y) );
        EXPECT_TRUE( all(z <= x) );
    }


    // int

    {
        I a( 1,  2,  3,   4,  5,  6,  7,  8);
        I b( 9, 10, 11,  12, 13, 14, 15, 16);
        I c( 1,  0,  3,  0,   5,  0,  7,  0);
        I x(std::numeric_limits<int>::max());
        I y(std::numeric_limits<int>::min());

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );
        EXPECT_TRUE( all(a <  b) );
        EXPECT_TRUE( all(a <= b) );
        EXPECT_TRUE( all(b >  a) );
        EXPECT_TRUE( all(b >= a) );
        EXPECT_TRUE( all(c <= a) );
        EXPECT_TRUE( all(a >= c) );

        EXPECT_TRUE( all((a > c) == M(0,1,0,1, 0,1,0,1)) );
        EXPECT_TRUE( all((c < a) == M(0,1,0,1, 0,1,0,1)) );

        EXPECT_TRUE( all(x >  I(0)) );
        EXPECT_TRUE( all(y <  I(0)) );
        EXPECT_TRUE( all(x >= I(0)) );
        EXPECT_TRUE( all(y <= I(0)) );

        EXPECT_TRUE( all(y <  x) );
        EXPECT_TRUE( all(y <= x) );
    }


    // mask

    {
        M a(0,0,0,0, 0,0,0,0);
        M b(1,1,1,1, 1,1,1,1);
        M c(1,0,1,0, 1,0,1,0);

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );

        EXPECT_TRUE( all((a == c) == M(0,1,0,1, 0,1,0,1)) );
        EXPECT_TRUE( all((a != c) == M(1,0,1,0, 1,0,1,0)) );
    }
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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
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


template <typename F>
static void test_math()
{
    // copysign(float)

    F xf(1.0f);
    F yf(2.0f);
    F rf = copysign(xf, yf);
    EXPECT_TRUE( all(rf == 1.0f) );

    xf = F(1.0f);
    yf = F(-2.0f);
    rf = copysign(xf, yf);
    EXPECT_TRUE( all(rf == -1.0f) );

    xf = F(INFINITY);
    yf = F(-2.0f);
    rf = copysign(xf, yf);
    EXPECT_TRUE( all(rf == -INFINITY) );

    // Other than the standard demands,
    // copysign(NAN, -y) will not return -NAN
    xf = F(NAN);
    yf = F(-2.0f);
    rf = copysign(xf, yf);
//  EXPECT_TRUE( all(rf == -NAN) );
}


//-------------------------------------------------------------------------------------------------
// Test simd::get()
//

TEST(SIMD, Get)
{
    // int4 -----------------------------------------------

    simd::int4 i4;
    iota4(i4);

    EXPECT_TRUE( simd::get<0>(i4) == 0 );
    EXPECT_TRUE( simd::get<1>(i4) == 1 );
    EXPECT_TRUE( simd::get<2>(i4) == 2 );
    EXPECT_TRUE( simd::get<3>(i4) == 3 );


    // float4 ---------------------------------------------

    simd::float4 f4;
    iota4(f4);

    EXPECT_FLOAT_EQ( simd::get<0>(f4), 0.0f );
    EXPECT_FLOAT_EQ( simd::get<1>(f4), 1.0f );
    EXPECT_FLOAT_EQ( simd::get<2>(f4), 2.0f );
    EXPECT_FLOAT_EQ( simd::get<3>(f4), 3.0f );

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // int8 -----------------------------------------------

    simd::int8 i8;
    iota8(i8);

    EXPECT_TRUE( simd::get<0>(i8) == 0 );
    EXPECT_TRUE( simd::get<1>(i8) == 1 );
    EXPECT_TRUE( simd::get<2>(i8) == 2 );
    EXPECT_TRUE( simd::get<3>(i8) == 3 );
    EXPECT_TRUE( simd::get<4>(i8) == 4 );
    EXPECT_TRUE( simd::get<5>(i8) == 5 );
    EXPECT_TRUE( simd::get<6>(i8) == 6 );
    EXPECT_TRUE( simd::get<7>(i8) == 7 );


    // float8 ---------------------------------------------

    simd::float8 f8;
    iota8(f8);

    EXPECT_FLOAT_EQ( simd::get<0>(f8), 0.0f );
    EXPECT_FLOAT_EQ( simd::get<1>(f8), 1.0f );
    EXPECT_FLOAT_EQ( simd::get<2>(f8), 2.0f );
    EXPECT_FLOAT_EQ( simd::get<3>(f8), 3.0f );
    EXPECT_FLOAT_EQ( simd::get<4>(f8), 4.0f );
    EXPECT_FLOAT_EQ( simd::get<5>(f8), 5.0f );
    EXPECT_FLOAT_EQ( simd::get<6>(f8), 6.0f );
    EXPECT_FLOAT_EQ( simd::get<7>(f8), 7.0f );

#endif
}


//-------------------------------------------------------------------------------------------------
// Test isnan(), isinf(), and isfinite()
//

TEST(SIMD, Representability)
{
    test_representability<simd::float4>();
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_representability<simd::float8>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test all() and any()
//

TEST(SIMD, Pred)
{
    test_pred_4();
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_pred_8();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test comparisons
//

TEST(SIMD, Comparison)
{
    test_cmp_4();
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_cmp_8();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test logical operations
//

TEST(SIMD, Logical)
{
    test_logical_4();
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_logical_8();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test math functions
//

TEST(SIMD, Math)
{
    test_math<simd::float4>();
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_math<simd::float8>();
#endif
}
