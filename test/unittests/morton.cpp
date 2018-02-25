// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/morton.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Morton, Encode)
{
    int z = morton_encode(0, 0, 0);
    ASSERT_EQ(z, 0);
    z = morton_encode(1, 0, 0);
    ASSERT_EQ(z, 1);
    z = morton_encode(0, 1, 0);
    ASSERT_EQ(z, 2);
    z = morton_encode(1, 1, 0);
    ASSERT_EQ(z, 3);

    z = morton_encode(0, 0, 1);
    ASSERT_EQ(z, 4);
    z = morton_encode(1, 0, 1);
    ASSERT_EQ(z, 5);
    z = morton_encode(0, 1, 1);
    ASSERT_EQ(z, 6);
    z = morton_encode(1, 1, 1);
    ASSERT_EQ(z, 7);
}

TEST(Morton, Decode)
{
    vec3i p = morton_decode(0);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 0);

    p = morton_decode(1);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 0);

    p = morton_decode(2);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 0);

    p = morton_decode(3);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 0);

    p = morton_decode(4);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 1);

    p = morton_decode(5);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 0);
    ASSERT_EQ(p.z, 1);

    p = morton_decode(6);
    ASSERT_EQ(p.x, 0);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 1);

    p = morton_decode(7);
    ASSERT_EQ(p.x, 1);
    ASSERT_EQ(p.y, 1);
    ASSERT_EQ(p.z, 1);
}
