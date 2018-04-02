// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>
#include <visionaray/phase_function.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test normalization property of phase functions
//

template <typename PF>
void test_normalization(PF const& pf)
{
    using T = typename PF::scalar_type;
    using Vec3 = vector<3, T>;


    // Test normalization property of phase function

    // Phase functions only depend on the angle theta
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
        int_ += pf.tr(-wo, wi) * sin(a) * d;

        wi.x = cos(a);
        wi.y = -sin(a);
        angle += deg;
    }

    int_ *= constants::two_pi<T>();

    T diff = T(1.0) - int_;

    ASSERT_TRUE(abs(diff) < T(1e-2));
}

TEST(PhaseFunction, Normalization)
{
    henyey_greenstein<float>  HGf;
    henyey_greenstein<double> HGd;

    for (double g = -0.9; g <= 0.9; g += 0.1)
    {
        HGf.g = g;
        HGd.g = g;

        test_normalization(HGf);
        test_normalization(HGd);
    }
}
