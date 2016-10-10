// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/sampling.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Sampling, RadicalInverse)
{
#define RI2(n) detail::radical_inverse<2>(n)
#define RI10(n) detail::radical_inverse<10>(n)

    EXPECT_FLOAT_EQ(RI2(1), 0.5f);
    EXPECT_FLOAT_EQ(RI2(2), 0.25f);
    EXPECT_FLOAT_EQ(RI2(4), 0.125f);
    EXPECT_FLOAT_EQ(RI2(8), 0.0625f);

    EXPECT_FLOAT_EQ(RI10(   1), 0.1f);
    EXPECT_FLOAT_EQ(RI10(   2), 0.2f);
    EXPECT_FLOAT_EQ(RI10(  23), 0.32f);
    EXPECT_FLOAT_EQ(RI10(4711), 0.1174f);
}
