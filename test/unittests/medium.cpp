// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <limits>

#include <visionaray/math/math.h>
#include <visionaray/medium.h>
#include <visionaray/random_generator.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test anisotropic participating medium
//

template <typename T>
void test_anisotropic(T g)
{
    using Vec3 = vector<3, T>;

    anisotropic_medium<T> am;
    am.anisotropy() = g;

    random_generator<T> rng;


    // Test sample()

    for (int i = 0; i < 100; ++i)
    {
        // Generate a random direction vector
        Vec3 wo = normalize(Vec3(rng.next(), rng.next(), rng.next()));

        // Sample light directions
        Vec3 wi;
        T pdf;
        am.sample(wo, wi, pdf, rng);

        // Test if light direction is normalized
        EXPECT_FLOAT_EQ(length(wi), T(1.0));
    }


    // Test the extreme cases that g is either -1.0 or 1.0

    for (int i = 0; i < 100; ++i)
    {
        // Generate a random direction vector
        Vec3 wo = normalize(Vec3(rng.next(), rng.next(), rng.next()));

        // Sample light directions
        Vec3 wi;
        T pdf;

        am.anisotropy() = T(-1.0);
        am.sample(wo, wi, pdf, rng);

        Vec3 v1 = wo - wi; // perfect backward scattering
        EXPECT_FLOAT_EQ(v1.x, 0.0f);
        EXPECT_FLOAT_EQ(v1.y, 0.0f);
        EXPECT_FLOAT_EQ(v1.z, 0.0f);

        am.anisotropy() = T(1.0);
        am.sample(wo, wi, pdf, rng);

        Vec3 v2 = wo + wi; // perfect forward scattering
        EXPECT_FLOAT_EQ(v2.x, 0.0f);
        EXPECT_FLOAT_EQ(v2.y, 0.0f);
        EXPECT_FLOAT_EQ(v2.z, 0.0f);
    }
}

TEST(Medium, Anisotropic)
{
    for (double g = -0.9; g <= 0.9; g += 0.1)
    {
        test_anisotropic<float>(g);
        test_anisotropic<double>(g);
    }
}
