// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <cstddef>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Helper functions
//

// nested for loop over matrices --------------------------

template <typename Func>
void for_each_mat2_e(Func f)
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            f(i, j);
        }
    }
}

template <typename Func>
void for_each_mat3_e(Func f)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            f(i, j);
        }
    }
}

template <typename Func>
void for_each_mat4_e(Func f)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            f(i, j);
        }
    }
}


// get rows and columns -----------------------------------

template <size_t Dim>
vector<Dim, float> get_row(matrix<Dim, Dim, float> const& m, int i)
{
    assert( i >= 0 && i < 4 );

    vector<Dim, float> result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = m(i, d);
    }

    return result;
}

template <size_t Dim>
vector<Dim, float> get_col(matrix<Dim, Dim, float> const& m, int j)
{
    assert( j >= 0 && j < 4 );

    return m(j);
}


TEST(Matrix, Inverse)
{

    //-------------------------------------------------------------------------
    // mat2
    //

    {

        mat2 I = mat2::identity();

        // make some non-singular matrix
        mat2 A(1, 2, 3, 4);
        mat2 B = inverse(A);
        mat2 C = A * B;

        for_each_mat2_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(C(i, j), I(i, j));
            }
            );

    }


    //-------------------------------------------------------------------------
    // mat3
    //

    {

        mat3 I = mat3::identity();

        // make some non-singular matrix
        mat3 A = mat3::rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
        mat3 B = inverse(A);
        mat3 C = A * B;

        for_each_mat3_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(C(i, j), I(i, j));
            }
            );

    }


    //-------------------------------------------------------------------------
    // mat4
    //

    {

        mat4 I = mat4::identity();

        // make some non-singular matrix
        mat4 A = mat4::rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
        mat4 B = inverse(A);
        mat4 C = A * B;

        for_each_mat4_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(C(i, j), I(i, j));
            }
            );

    }
}

TEST(Matrix, Mul)
{

    //-------------------------------------------------------------------------
    // mat2
    //

    {

        // make some matrices
        mat2 A = mat2::identity(); A(0, 0) = 2.0f;  A(1, 0) = 3.14f; A(1, 1) = 3.0f;
        mat2 B = mat2::identity(); B(0, 0) = 11.0f; B(1, 0) = 6.28f; B(1, 1) = 3.0f;
        mat2 C = A * B;

        for_each_mat2_e(
            [&](int i, int j)
            {
                float d = dot(get_row(A, i), get_col(B, j));
                EXPECT_FLOAT_EQ(C(i, j), d);
            }
            );
    }


    //-------------------------------------------------------------------------
    // mat3
    //

    {

        // make some matrices
        mat3 A = mat3::identity(); A(0, 0) = 2.0f;  A(1, 0) = 3.14f; A(1, 1) = 3.0f;
        mat3 B = mat3::identity(); B(0, 1) = 11.0f; B(2, 1) = 6.28f; B(2, 2) = 3.0f;
        mat3 C = A * B;

        for_each_mat3_e(
            [&](int i, int j)
            {
                float d = dot(get_row(A, i), get_col(B, j));
                EXPECT_FLOAT_EQ(C(i, j), d);
            }
            );
    }


    //-------------------------------------------------------------------------
    // mat4
    //

    {

        // make some matrices
        mat4 A = mat4::rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
        mat4 B = mat4::translation(vec3(3, 4, 5));
        mat4 C = A * B;

        for_each_mat4_e(
            [&](int i, int j)
            {
                float d = dot(get_row(A, i), get_col(B, j));
                EXPECT_FLOAT_EQ(C(i, j), d);
            }
            );

    }
}

TEST(Matrix, Add)
{
    //-------------------------------------------------------------------------
    // mat4
    //

    {

        // make some matrices
        mat4 A = mat4::rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
        mat4 B = mat4::translation(vec3(3, 4, 5));
        mat4 C = A + B;

        for_each_mat4_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(C(i, j), A(i, j) + B(i, j));
            }
            );

    }
}

TEST(Matrix, Transpose)
{

    //-------------------------------------------------------------------------
    // mat2
    //

    {

        // make some non-singular matrix
        mat2 A = mat2::identity(); A(0, 0) = 2.0f;  A(1, 0) = 3.14f; A(1, 1) = 3.0f;
        mat2 B = transpose(A);

        for_each_mat2_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(A(i, j), B(j, i));
            }
            );

    }


    //-------------------------------------------------------------------------
    // mat3
    //

    {

        // make some non-singular matrix
        mat3 A = mat3::identity(); A(0, 0) = 2.0f;  A(1, 0) = 3.14f; A(1, 1) = 3.0f;
        mat3 B = transpose(A);

        for_each_mat3_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(A(i, j), B(j, i));
            }
            );

    }


    //-------------------------------------------------------------------------
    // mat4
    //

    {

        // make some non-singular matrix
        mat4 A = mat4::rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
        mat4 B = transpose(A);

        for_each_mat4_e(
            [&](int i, int j)
            {
                EXPECT_FLOAT_EQ(A(i, j), B(j, i));
            }
            );

    }
}
