// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <limits>

#include <visionaray/math/math.h>
#include <visionaray/medium.h>
#include <visionaray/random_sampler.h>

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

    random_sampler<T> rs;


    // Test sample()

    for (int i = 0; i < 100; ++i)
    {
        // Generate a random direction vector
        Vec3 wo = normalize(Vec3(rs.next(), rs.next(), rs.next()));

        // Sample light directions
        Vec3 wi;
        T pdf;
        am.sample(wo, wi, pdf, rs);

        // Test if light direction is normalized
        EXPECT_FLOAT_EQ(length(wi), T(1.0));
    }


    // Test normalization property of HG phase function

    // HG phase function only depends on the angle theta
    // between wi and wo, so we can choose an arbitrary
    // integration plane

    Vec3 wo(-1.0, 0.0, 0.0);
    Vec3 wi( 1.0, 0.0, 0.0);

    // Just to make sure..
    EXPECT_FLOAT_EQ(dot(wo, wi), -1.0f);

    T angle(0.0);
    T deg(0.001); // integrate in deg steps
    T d = deg * constants::degrees_to_radians<T>();

    T int_(0.0);

    while (angle <= T(180.0))
    {
        T a = angle * constants::degrees_to_radians<T>();

        // Riemann integration
        int_ += am.tr(-wo, wi)[0] * sin(a) * d;

        wi.x = cos(a);
        wi.y = -sin(a);
        angle += deg;
    }

    int_ *= constants::two_pi<T>();

    T diff = T(1.0) - int_;

    ASSERT_TRUE(abs(diff) < T(1e-2));


    // Test the extreme cases that g is either -1.0 or 1.0

    for (int i = 0; i < 100; ++i)
    {
        // Generate a random direction vector
        Vec3 wo = normalize(Vec3(rs.next(), rs.next(), rs.next()));

        // Sample light directions
        Vec3 wi;
        T pdf;

        am.anisotropy() = T(-1.0);
        am.sample(wo, wi, pdf, rs);

        Vec3 v1 = wo - wi; // perfect backward scattering
        EXPECT_FLOAT_EQ(v1.x, 0.0f);
        EXPECT_FLOAT_EQ(v1.y, 0.0f);
        EXPECT_FLOAT_EQ(v1.z, 0.0f);

        am.anisotropy() = T(1.0);
        am.sample(wo, wi, pdf, rs);

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
