// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/morton.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Morton, Encode2D)
{
    int z;

    z = morton_encode2D(0, 0);
    ASSERT_EQ(z, 0);
    z = morton_encode2D(1, 0);
    ASSERT_EQ(z, 1);
    z = morton_encode2D(0, 1);
    ASSERT_EQ(z, 2);
    z = morton_encode2D(1, 1);
    ASSERT_EQ(z, 3);

    z = morton_encode2D(2, 0);
    ASSERT_EQ(z, 4);
    z = morton_encode2D(3, 0);
    ASSERT_EQ(z, 5);
    z = morton_encode2D(2, 1);
    ASSERT_EQ(z, 6);
    z = morton_encode2D(3, 1);
    ASSERT_EQ(z, 7);

    z = morton_encode2D(0, 2);
    ASSERT_EQ(z, 8);
    z = morton_encode2D(1, 2);
    ASSERT_EQ(z, 9);
    z = morton_encode2D(0, 3);
    ASSERT_EQ(z, 10);
    z = morton_encode2D(1, 3);
    ASSERT_EQ(z, 11);
}

TEST(Morton, Encode3D)
{
    int z;

    z = morton_encode3D(0, 0, 0);
    ASSERT_EQ(z, 0);
    z = morton_encode3D(1, 0, 0);
    ASSERT_EQ(z, 1);
    z = morton_encode3D(0, 1, 0);
    ASSERT_EQ(z, 2);
    z = morton_encode3D(1, 1, 0);
    ASSERT_EQ(z, 3);

    z = morton_encode3D(0, 0, 1);
    ASSERT_EQ(z, 4);
    z = morton_encode3D(1, 0, 1);
    ASSERT_EQ(z, 5);
    z = morton_encode3D(0, 1, 1);
    ASSERT_EQ(z, 6);
    z = morton_encode3D(1, 1, 1);
    ASSERT_EQ(z, 7);
}

TEST(Morton, Decode2D)
{
    vec2i p;

    p = morton_decode2D(0);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 0);

    p = morton_decode2D(1);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 0);

    p = morton_decode2D(2);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 1);

    p = morton_decode2D(3);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 1);

    p = morton_decode2D(4);
    ASSERT_EQ(p.x, 2);
    ASSERT_EQ(p.y, 0);

    p = morton_decode2D(5);
    ASSERT_EQ(p.x, 3);
    ASSERT_EQ(p.y, 0);

    p = morton_decode2D(6);
    ASSERT_EQ(p.x, 2);
    ASSERT_EQ(p.y, 1);

    p = morton_decode2D(7);
    ASSERT_EQ(p.x, 3);
    ASSERT_EQ(p.y, 1);

    p = morton_decode2D(8);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 2);

    p = morton_decode2D(9);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 2);

    p = morton_decode2D(10);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 3);

    p = morton_decode2D(11);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 3);
}

TEST(Morton, Decode3D)
{
    vec3i p;

    p = morton_decode3D(0);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 0);

    p = morton_decode3D(1);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 0);

    p = morton_decode3D(2);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 0);

    p = morton_decode3D(3);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 0);

    p = morton_decode3D(4);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 1);

    p = morton_decode3D(5);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 1);

    p = morton_decode3D(6);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 1);

    p = morton_decode3D(7);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 1);
}
