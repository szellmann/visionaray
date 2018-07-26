// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/triangle.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Triangle, Area)
{
    vec3 v1(0, 0, 0);
    vec3 v2(2, 0, 0);
    vec3 v3(2, 2, 0);
    basic_triangle<3, float> tri(
        v1,
        v2 - v1,
        v3 - v1
        );

    EXPECT_FLOAT_EQ(area(tri), 2.0f);

    v1 = vec3(0, 0, 0);
    v2 = vec3(2, 0, 0);
    v3 = vec3(1, 2, 0);
    tri = basic_triangle<3, float>(
        v1,
        v2 - v1,
        v3 - v1
        );

    EXPECT_FLOAT_EQ(area(tri), 2.0f);

    v1 = vec3(0, 0, 0);
    v2 = vec3(2, 0, 2);
    v3 = vec3(1, 2, 1);
    tri = basic_triangle<3, float>(
        v1,
        v2 - v1,
        v3 - v1
        );

    EXPECT_FLOAT_EQ(area(tri), 2.8284271f);
}
