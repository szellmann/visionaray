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
