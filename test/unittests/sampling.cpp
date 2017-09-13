// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/simd/simd.h>
#include <visionaray/random_sampler.h>
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

TEST(Sampling, ConcentricSampleDisk)
{
    static const int NumSamples = 10000;

    random_sampler<float> rs1(0U);
    random_sampler<simd::float4> rs4(0U);
    random_sampler<simd::float8> rs8(0U);

    // Check that all samples are inside the disk
    for (int i = 0; i < NumSamples; ++i)
    {
        float u1 = rs1.next();
        float u2 = rs1.next();

        auto sample = concentric_sample_disk(u1, u2);
        EXPECT_TRUE(length(sample) >= 0.0f);
        EXPECT_TRUE(length(sample) <= 1.0f);
    }

    for (int i = 0; i < NumSamples / 4; ++i)
    {
        simd::float4 u1 = rs4.next();
        simd::float4 u2 = rs4.next();

        auto sample = concentric_sample_disk(u1, u2);
        EXPECT_TRUE(all(length(sample) >= simd::float4(0.0f)));
        EXPECT_TRUE(all(length(sample) <= simd::float4(1.0f)));
    }

    for (int i = 0; i < NumSamples / 8; ++i)
    {
        simd::float8 u1 = rs8.next();
        simd::float8 u2 = rs8.next();

        auto sample = concentric_sample_disk(u1, u2);
        EXPECT_TRUE(all(length(sample) >= simd::float8(0.0f)));
        EXPECT_TRUE(all(length(sample) <= simd::float8(1.0f)));
    }
}
