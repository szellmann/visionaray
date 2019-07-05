// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/morton.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Morton, Encode2D)
{
    unsigned z;

    z = morton_encode2D(0, 0);
    EXPECT_EQ(z, 0);
    z = morton_encode2D(1, 0);
    EXPECT_EQ(z, 1);
    z = morton_encode2D(0, 1);
    EXPECT_EQ(z, 2);
    z = morton_encode2D(1, 1);
    EXPECT_EQ(z, 3);

    z = morton_encode2D(2, 0);
    EXPECT_EQ(z, 4);
    z = morton_encode2D(3, 0);
    EXPECT_EQ(z, 5);
    z = morton_encode2D(2, 1);
    EXPECT_EQ(z, 6);
    z = morton_encode2D(3, 1);
    EXPECT_EQ(z, 7);

    z = morton_encode2D(0, 2);
    EXPECT_EQ(z, 8);
    z = morton_encode2D(1, 2);
    EXPECT_EQ(z, 9);
    z = morton_encode2D(0, 3);
    EXPECT_EQ(z, 10);
    z = morton_encode2D(1, 3);
    EXPECT_EQ(z, 11);

    // Diagonal, 2^0..2^15
    for (unsigned i = 0; i < 15; ++i)
    {
        z = morton_encode2D(2 << i, 2 << i);
        EXPECT_EQ(z, (2 << i) * (2 << i) * 3);
    }
}

TEST(Morton, Encode3D)
{
    unsigned z;

    z = morton_encode3D(0, 0, 0);
    EXPECT_EQ(z, 0);
    z = morton_encode3D(1, 0, 0);
    EXPECT_EQ(z, 1);
    z = morton_encode3D(0, 1, 0);
    EXPECT_EQ(z, 2);
    z = morton_encode3D(1, 1, 0);
    EXPECT_EQ(z, 3);

    z = morton_encode3D(0, 0, 1);
    EXPECT_EQ(z, 4);
    z = morton_encode3D(1, 0, 1);
    EXPECT_EQ(z, 5);
    z = morton_encode3D(0, 1, 1);
    EXPECT_EQ(z, 6);
    z = morton_encode3D(1, 1, 1);
    EXPECT_EQ(z, 7);
}

TEST(Morton, Decode2D)
{
    vec2ui p;

    p = morton_decode2D(0);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 0);

    p = morton_decode2D(1);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 0);

    p = morton_decode2D(2);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 1);

    p = morton_decode2D(3);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 1);

    p = morton_decode2D(4);
    EXPECT_EQ(p.x, 2);
    EXPECT_EQ(p.y, 0);

    p = morton_decode2D(5);
    EXPECT_EQ(p.x, 3);
    EXPECT_EQ(p.y, 0);

    p = morton_decode2D(6);
    EXPECT_EQ(p.x, 2);
    EXPECT_EQ(p.y, 1);

    p = morton_decode2D(7);
    EXPECT_EQ(p.x, 3);
    EXPECT_EQ(p.y, 1);

    p = morton_decode2D(8);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 2);

    p = morton_decode2D(9);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 2);

    p = morton_decode2D(10);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 3);

    p = morton_decode2D(11);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 3);
}

TEST(Morton, Decode3D)
{
    vec3ui p;

    p = morton_decode3D(0);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 0);
    EXPECT_EQ(p.z, 0);

    p = morton_decode3D(1);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 0);
    EXPECT_EQ(p.z, 0);

    p = morton_decode3D(2);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 1);
    EXPECT_EQ(p.z, 0);

    p = morton_decode3D(3);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 1);
    EXPECT_EQ(p.z, 0);

    p = morton_decode3D(4);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 0);
    EXPECT_EQ(p.z, 1);

    p = morton_decode3D(5);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 0);
    EXPECT_EQ(p.z, 1);

    p = morton_decode3D(6);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 1);
    EXPECT_EQ(p.z, 1);

    p = morton_decode3D(7);
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 1);
    EXPECT_EQ(p.z, 1);
}
