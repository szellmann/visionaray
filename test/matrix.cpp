// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Helper functions
//

// nested for loop over matrices --------------------------

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

vec4 get_row(mat4 const& m, int i)
{
    assert( i >= 0 && i < 4 );

    return vec4(
        m(i, 0),
        m(i, 1),
        m(i, 2),
        m(i, 3)
        );
}

vec4 get_col(mat4 const& m, int j)
{
    assert( j >= 0 && j < 4 );

    return m(j);
}


TEST(Matrix, Inverse)
{

    //-------------------------------------------------------------------------
    // mat4
    //

    mat4 I = mat4::identity();

    // make some non-singular matrix
    mat4 A = make_rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
    mat4 B = inverse(A);
    mat4 C = A * B;

    for_each_mat4_e(
        [&](int i, int j)
        {
            EXPECT_FLOAT_EQ(C(i, j), I(i, j));
        }
        );
}

TEST(Matrix, Mult)
{

    //-------------------------------------------------------------------------
    // mat4
    //

    // make some non-singular matrices
    mat4 A = make_rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
    mat4 B = make_translation(vec3(3, 4, 5));
    mat4 C = A * B;

    for_each_mat4_e(
        [&](int i, int j)
        {
            float d = dot(get_row(A, i), get_col(B, j));
            EXPECT_FLOAT_EQ(C(i, j), d);
        }
        );
}

TEST(Matrix, Transpose)
{

    //-------------------------------------------------------------------------
    // mat4
    //

    // make some non-singular matrix
    mat4 A = make_rotation(vec3(1, 0, 0), constants::pi<float>() / 4);
    mat4 B = transpose(A);

    for_each_mat4_e(
        [&](int i, int j)
        {
            EXPECT_FLOAT_EQ(A(i, j), B(j, i));
        }
        );
}
