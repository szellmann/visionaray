// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cfloat>
#include <initializer_list>
#include <limits>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
//
//

template <typename T>
inline T const* first_of(std::initializer_list<T> il)
{
    return il.begin();
}


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

template <typename T>
static void iota16(T& t)
{
    simd::get< 0>(t) =  0;
    simd::get< 1>(t) =  1;
    simd::get< 2>(t) =  2;
    simd::get< 3>(t) =  3;
    simd::get< 4>(t) =  4;
    simd::get< 5>(t) =  5;
    simd::get< 6>(t) =  6;
    simd::get< 7>(t) =  7;
    simd::get< 8>(t) =  8;
    simd::get< 9>(t) =  9;
    simd::get<10>(t) = 10;
    simd::get<11>(t) = 11;
    simd::get<12>(t) = 12;
    simd::get<13>(t) = 13;
    simd::get<14>(t) = 14;
    simd::get<15>(t) = 15;
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

template <typename F, typename I, typename M>
static void test_pred()
{
    M z(first_of<bool>({0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0}));
    M a(first_of<bool>({1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0}));
    M i(first_of<bool>({1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1}));

    EXPECT_TRUE(!any(z) );
    EXPECT_TRUE(!all(z) );
    EXPECT_TRUE( any(a) );
    EXPECT_TRUE(!all(a) );
    EXPECT_TRUE( any(i) );
    EXPECT_TRUE( all(i) );
}

template <typename F, typename I, typename M>
static void test_cmp()
{
    // float

    {
        F a(first_of({
             1.0f,  2.0f,  3.0f,  4.0f,   5.0f,  6.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 11.0f, 12.0f,  13.0f, 14.0f, 15.0f, 16.0f
            }));
        F b(first_of({
            17.0f, 18.0f, 19.0f, 20.0f,  21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f,  29.0f, 30.0f, 31.0f, 32.0f
            }));
        F c(first_of({
             1.0f,  0.0f,  3.0f,  0.0f,   5.0f,  0.0f,  7.0f,  0.0f,
             9.0f,  0.0f, 11.0f,  0.0f,  13.0f,  0.0f, 15.0f,  0.0f
            }));
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

        EXPECT_TRUE( all((a > c) == M(first_of<bool>({0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1}))) );
        EXPECT_TRUE( all((c < a) == M(first_of<bool>({0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1}))) );

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
        I a(first_of({ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16}));
        I b(first_of({17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}));
        I c(first_of({ 1,  0,  3,  0,  5,  0,  7,  0,  9,  0, 11,  0, 13,  0, 15,  0}));
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

        EXPECT_TRUE( all((a > c) == M(first_of<bool>({0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1}))) );
        EXPECT_TRUE( all((c < a) == M(first_of<bool>({0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1}))) );

        EXPECT_TRUE( all(x >  I(0)) );
        EXPECT_TRUE( all(y <  I(0)) );
        EXPECT_TRUE( all(x >= I(0)) );
        EXPECT_TRUE( all(y <= I(0)) );

        EXPECT_TRUE( all(y <  x) );
        EXPECT_TRUE( all(y <= x) );
    }


    // mask

    {
        M a(first_of<bool>({0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0}));
        M b(first_of<bool>({1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1}));
        M c(first_of<bool>({1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0}));

        EXPECT_TRUE( all(a == a) );
        EXPECT_TRUE( all(a != b) );

        EXPECT_TRUE( all((a == c) == M(first_of<bool>({0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1}))) );
        EXPECT_TRUE( all((a != c) == M(first_of<bool>({1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0}))) );
    }
}

template <typename F, typename I, typename M>
static void test_logical()
{
    // mask

    {
        M a(first_of<bool>({1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0}));
        M b(first_of<bool>({1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0}));
        M c(first_of<bool>({0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1}));

        EXPECT_TRUE( all((a && b) == M(first_of<bool>({1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0}))) );
        EXPECT_TRUE( all((a && c) == M(first_of<bool>({0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0}))) );
        EXPECT_TRUE( all((a || b) == M(first_of<bool>({1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0}))) );
        EXPECT_TRUE( all((a || c) == M(first_of<bool>({1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1}))) );

        EXPECT_TRUE( any(a && b) );
        EXPECT_TRUE(!any(a && c) );
        EXPECT_TRUE( any(a || b) );
        EXPECT_TRUE( all(a || c) );

        EXPECT_TRUE( any(!(a && b)) );
        EXPECT_TRUE( all(!(a && c)) );
        EXPECT_TRUE( any(!(a || b)) );
        EXPECT_TRUE(!any(!(a || c)) );
    }
}

template <typename F>
static void test_math()
{
    using I = simd::int_type_t<F>;

    // basic arithmetic

    {
        // float
        VSNRAY_ALIGN(64) float af_arr[] = {
                 0.0f,  1.0f,  2.0f,  3.0f,
                 4.0f,  5.0f,  6.0f,  7.0f,
                 8.0f,  9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f
                };
        F af(af_arr);
        F bf(2.0);

        EXPECT_TRUE( all(+af     == F(0.0) + af) );
        EXPECT_TRUE( all(-af     == F(0.0) - af) );
        EXPECT_TRUE( all(af + af == af * F(2.0)) );
        EXPECT_TRUE( all(af - af == F(0.0)) );
//      EXPECT_TRUE( all(af * af == pow(af, F(2.0))) );
        EXPECT_TRUE( all(af / bf == af * F(0.5)) );


        // int
        VSNRAY_ALIGN(64) int ai_arr[] = {
                 0,  1,  2,  3,
                 4,  5,  6,  7,
                 8,  9, 10, 11,
                12, 13, 14, 15
                };
        I ai(ai_arr);
        I bi(2);

        EXPECT_TRUE( all(+ai     == I(0) + ai) );
        EXPECT_TRUE( all(-ai     == I(0) - ai) );
        EXPECT_TRUE( all(ai + ai == ai * I(2)) );
        EXPECT_TRUE( all(ai - ai == I(0)) );
    }

    // modulo

    {
        EXPECT_TRUE( all(I( 0) % I( 2) == I( 0)) );
        EXPECT_TRUE( all(I( 1) % I( 2) == I( 1)) );
        EXPECT_TRUE( all(I( 2) % I( 2) == I( 0)) );
        EXPECT_TRUE( all(I( 3) % I( 2) == I( 1)) );
        EXPECT_TRUE( all(I( 4) % I( 2) == I( 0)) );
        EXPECT_TRUE( all(I( 5) % I( 2) == I( 1)) );

        EXPECT_TRUE( all(I(-0) % I( 2) == I(-0)) );
        EXPECT_TRUE( all(I(-1) % I( 2) == I(-1)) );
        EXPECT_TRUE( all(I(-2) % I( 2) == I(-0)) );
        EXPECT_TRUE( all(I(-3) % I( 2) == I(-1)) );
        EXPECT_TRUE( all(I(-4) % I( 2) == I(-0)) );
        EXPECT_TRUE( all(I(-5) % I( 2) == I(-1)) );

        EXPECT_TRUE( all(I( 0) % I(-2) == I( 0)) );
        EXPECT_TRUE( all(I( 1) % I(-2) == I( 1)) );
        EXPECT_TRUE( all(I( 2) % I(-2) == I( 0)) );
        EXPECT_TRUE( all(I( 3) % I(-2) == I( 1)) );
        EXPECT_TRUE( all(I( 4) % I(-2) == I( 0)) );
        EXPECT_TRUE( all(I( 5) % I(-2) == I( 1)) );

        EXPECT_TRUE( all(I(-0) % I(-2) == I(-0)) );
        EXPECT_TRUE( all(I(-1) % I(-2) == I(-1)) );
        EXPECT_TRUE( all(I(-2) % I(-2) == I(-0)) );
        EXPECT_TRUE( all(I(-3) % I(-2) == I(-1)) );
        EXPECT_TRUE( all(I(-4) % I(-2) == I(-0)) );
        EXPECT_TRUE( all(I(-5) % I(-2) == I(-1)) );
    }


    // misc math functions

    {
        F flow  = numeric_limits<F>::lowest();
        F fmin  = numeric_limits<F>::min();
        F fmax  = numeric_limits<F>::max();
        F fzero = 0.0f;
        F fp    = 23.0f;
        F fn    = -23.0f;

        EXPECT_TRUE( all(round(F(3.14f)) == F(3.0f)) );
        EXPECT_TRUE( all(round(F(0.6f))  == F(1.0f)) );
        EXPECT_TRUE( all(round(F(-0.1f)) == F(0.0f)) );
        EXPECT_TRUE( all(round(F(-0.6f)) == F(-1.0f)) );

        EXPECT_TRUE( all(ceil(flow)      == ceil(numeric_limits<float>::lowest())) );
        EXPECT_TRUE( all(ceil(fmin)      == ceil(numeric_limits<float>::min())) );
        EXPECT_TRUE( all(ceil(fmax)      == ceil(numeric_limits<float>::max())) );
        EXPECT_TRUE( all(ceil(fzero)     == ceil(0.0f)) );
        EXPECT_TRUE( all(ceil(fp)        == 23.0f) );
        EXPECT_TRUE( all(ceil(fn)        == -23.0f) );

        EXPECT_TRUE( all(floor(flow)     == floor(numeric_limits<float>::lowest())) );
        EXPECT_TRUE( all(floor(fmin)     == floor(numeric_limits<float>::min())) );
        EXPECT_TRUE( all(floor(fmax)     == floor(numeric_limits<float>::max())) );
        EXPECT_TRUE( all(floor(fzero)    == floor(0.0f)) );
        EXPECT_TRUE( all(floor(fp)       == 23.0f) );
        EXPECT_TRUE( all(floor(fn)       == -23.0f) );

        EXPECT_TRUE( all(saturate(flow)  == 0.0f) );
        EXPECT_TRUE( all(saturate(fmin)  == numeric_limits<float>::min()) );
        EXPECT_TRUE( all(saturate(fmax)  == 1.0f) );
        EXPECT_TRUE( all(saturate(fzero) == 0.0f) );
        EXPECT_TRUE( all(saturate(fp)    == 1.0f) );
        EXPECT_TRUE( all(saturate(fn)    == 0.0f) );

        EXPECT_TRUE( all(sqrt(fmax)      == sqrt(numeric_limits<float>::max())) );
        EXPECT_TRUE( all(sqrt(fzero)     == sqrt(0.0f)) );
        EXPECT_TRUE( all(sqrt(fp)        == sqrt(23.0f)) );
    }


    // copysign(float)

    {
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
//      EXPECT_TRUE( all(rf == -NAN) );
    }


    // dot() (horizontal)

    {
        simd::float4 xf(1.0f, 2.0f, 3.0f, 4.0f);
        simd::float4 yf(5.0f, 6.0f, 7.0f, 8.0f);
        EXPECT_TRUE( all(dot(xf, yf) == simd::float4(70.0f)) );

        xf *= simd::float4(-1.0f);
        EXPECT_TRUE( all(dot(xf, yf) == simd::float4(-70.0f)) );
    }
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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)

    // int16 ----------------------------------------------

    simd::int16 i16;
    iota16(i16);

    EXPECT_TRUE( simd::get< 0>(i16) ==  0 );
    EXPECT_TRUE( simd::get< 1>(i16) ==  1 );
    EXPECT_TRUE( simd::get< 2>(i16) ==  2 );
    EXPECT_TRUE( simd::get< 3>(i16) ==  3 );
    EXPECT_TRUE( simd::get< 4>(i16) ==  4 );
    EXPECT_TRUE( simd::get< 5>(i16) ==  5 );
    EXPECT_TRUE( simd::get< 6>(i16) ==  6 );
    EXPECT_TRUE( simd::get< 7>(i16) ==  7 );
    EXPECT_TRUE( simd::get< 8>(i16) ==  8 );
    EXPECT_TRUE( simd::get< 9>(i16) ==  9 );
    EXPECT_TRUE( simd::get<10>(i16) == 10 );
    EXPECT_TRUE( simd::get<11>(i16) == 11 );
    EXPECT_TRUE( simd::get<12>(i16) == 12 );
    EXPECT_TRUE( simd::get<13>(i16) == 13 );
    EXPECT_TRUE( simd::get<14>(i16) == 14 );
    EXPECT_TRUE( simd::get<15>(i16) == 15 );


    // float16 --------------------------------------------

    simd::float16 f16;
    iota16(f16);

    EXPECT_FLOAT_EQ( simd::get< 0>(f16),  0.0f );
    EXPECT_FLOAT_EQ( simd::get< 1>(f16),  1.0f );
    EXPECT_FLOAT_EQ( simd::get< 2>(f16),  2.0f );
    EXPECT_FLOAT_EQ( simd::get< 3>(f16),  3.0f );
    EXPECT_FLOAT_EQ( simd::get< 4>(f16),  4.0f );
    EXPECT_FLOAT_EQ( simd::get< 5>(f16),  5.0f );
    EXPECT_FLOAT_EQ( simd::get< 6>(f16),  6.0f );
    EXPECT_FLOAT_EQ( simd::get< 7>(f16),  7.0f );
    EXPECT_FLOAT_EQ( simd::get< 8>(f16),  8.0f );
    EXPECT_FLOAT_EQ( simd::get< 9>(f16),  9.0f );
    EXPECT_FLOAT_EQ( simd::get<10>(f16), 10.0f );
    EXPECT_FLOAT_EQ( simd::get<11>(f16), 11.0f );
    EXPECT_FLOAT_EQ( simd::get<12>(f16), 12.0f );
    EXPECT_FLOAT_EQ( simd::get<13>(f16), 13.0f );
    EXPECT_FLOAT_EQ( simd::get<14>(f16), 14.0f );
    EXPECT_FLOAT_EQ( simd::get<15>(f16), 15.0f );

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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    test_representability<simd::float16>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test all() and any()
//

TEST(SIMD, Pred)
{
    test_pred<simd::float4, simd::int4, simd::mask4>();

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_pred<simd::float8, simd::int8, simd::mask8>();
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    test_pred<simd::float16, simd::int16, simd::mask16>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test comparisons
//

TEST(SIMD, Comparison)
{
    test_cmp<simd::float4, simd::int4, simd::mask4>();

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_cmp<simd::float8, simd::int8, simd::mask8>();
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    test_cmp<simd::float16, simd::int16, simd::mask16>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test logical operations
//

TEST(SIMD, Logical)
{
    test_logical<simd::float4, simd::int4, simd::mask4>();

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    test_logical<simd::float8, simd::int8, simd::mask8>();
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    test_logical<simd::float16, simd::int16, simd::mask16>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Test transposition operations (TODO: all ISA)
//

TEST(SIMD, Transposition)
{
    // shuffle

    simd::float4 u(1.0f, 2.0f, 3.0f, 4.0f);
    simd::float4 v(5.0f, 6.0f, 7.0f, 8.0f);

    EXPECT_TRUE( all(simd::shuffle<0, 1, 2, 3>(u)    == simd::float4(1.0f, 2.0f, 3.0f, 4.0f)) );
    EXPECT_TRUE( all(simd::shuffle<3, 2, 1, 0>(u)    == simd::float4(4.0f, 3.0f, 2.0f, 1.0f)) );
    EXPECT_TRUE( all(simd::shuffle<0, 0, 3, 3>(u)    == simd::float4(1.0f, 1.0f, 4.0f, 4.0f)) );
    EXPECT_TRUE( all(simd::shuffle<3, 3, 0, 0>(u)    == simd::float4(4.0f, 4.0f, 1.0f, 1.0f)) );
    EXPECT_TRUE( all(simd::shuffle<0, 1, 2, 3>(u, v) == simd::float4(1.0f, 2.0f, 7.0f, 8.0f)) );
    EXPECT_TRUE( all(simd::shuffle<3, 2, 1, 0>(u, v) == simd::float4(4.0f, 3.0f, 6.0f, 5.0f)) );
    EXPECT_TRUE( all(simd::shuffle<0, 0, 3, 3>(u, v) == simd::float4(1.0f, 1.0f, 8.0f, 8.0f)) );
    EXPECT_TRUE( all(simd::shuffle<3, 3, 0, 0>(u, v) == simd::float4(4.0f, 4.0f, 5.0f, 5.0f)) );


    // move_xx

    EXPECT_TRUE( all(move_lo(u, v)                   == simd::float4(1.0f, 2.0f, 5.0f, 6.0f)) );
    EXPECT_TRUE( all(move_hi(u, v)                   == simd::float4(7.0f, 8.0f, 3.0f, 4.0f)) );


    // interleave_xx

    EXPECT_TRUE( all(interleave_lo(u, v)             == simd::float4(1.0f, 5.0f, 2.0f, 6.0f)) );
    EXPECT_TRUE( all(interleave_hi(u, v)             == simd::float4(3.0f, 7.0f, 4.0f, 8.0f)) );
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

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    test_math<simd::float16>();
#endif
}


//-------------------------------------------------------------------------------------------------
// Transpose vector<4, float4> (SoA to AoS and vice versa)
//

TEST(SIMD, TransposeVec4)
{
    vector<4, simd::float4> v(
            simd::float4( 0.0f,  1.0f,  2.0f,  3.0f),
            simd::float4( 4.0f,  5.0f,  6.0f,  7.0f),
            simd::float4( 8.0f,  9.0f, 10.0f, 11.0f),
            simd::float4(12.0f, 13.0f, 14.0f, 15.0f)
            );

    vector<4, simd::float4> vt = transpose(v);
    EXPECT_TRUE( all(vt.x == simd::float4( 0.0f,  4.0f,  8.0f, 12.0f)) );
    EXPECT_TRUE( all(vt.y == simd::float4( 1.0f,  5.0f,  9.0f, 13.0f)) );
    EXPECT_TRUE( all(vt.z == simd::float4( 2.0f,  6.0f, 10.0f, 14.0f)) );
    EXPECT_TRUE( all(vt.w == simd::float4( 3.0f,  7.0f, 11.0f, 15.0f)) );
}
