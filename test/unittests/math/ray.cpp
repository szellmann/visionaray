// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


TEST(Ray, SIMDPack)
{
    // SSE ------------------------------------------------

    {
        std::array<basic_ray<float>, 4> rays;

        for (size_t i = 0; i < 4; ++i)
        {
            rays[i] = basic_ray<float>(
                    vec3(1.0f, 2.0f, 3.0f),
                    vec3(1.0f, 0.0f, 0.0f)
                    );
        }

        auto ray4 = simd::pack(rays);

        EXPECT_FLOAT_EQ(simd::get<0>(ray4.ori.x), rays[0].ori.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray4.ori.y), rays[0].ori.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray4.ori.z), rays[0].ori.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.ori.x), rays[1].ori.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.ori.y), rays[1].ori.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.ori.z), rays[1].ori.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.ori.x), rays[2].ori.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.ori.y), rays[2].ori.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.ori.z), rays[2].ori.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.ori.x), rays[3].ori.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.ori.y), rays[3].ori.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.ori.z), rays[3].ori.z);

        EXPECT_FLOAT_EQ(simd::get<0>(ray4.dir.x), rays[0].dir.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray4.dir.y), rays[0].dir.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray4.dir.z), rays[0].dir.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.dir.x), rays[1].dir.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.dir.y), rays[1].dir.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray4.dir.z), rays[1].dir.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.dir.x), rays[2].dir.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.dir.y), rays[2].dir.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray4.dir.z), rays[2].dir.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.dir.x), rays[3].dir.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.dir.y), rays[3].dir.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray4.dir.z), rays[3].dir.z);
    }

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

    // AVX ------------------------------------------------

    {
        std::array<basic_ray<float>, 8> rays;

        for (size_t i = 0; i < 8; ++i)
        {
            rays[i] = basic_ray<float>(
                    vec3(1.0f, 2.0f, 3.0f),
                    vec3(1.0f, 0.0f, 0.0f)
                    );
        }

        auto ray8 = simd::pack(rays);

        EXPECT_FLOAT_EQ(simd::get<0>(ray8.ori.x), rays[0].ori.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray8.ori.y), rays[0].ori.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray8.ori.z), rays[0].ori.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.ori.x), rays[1].ori.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.ori.y), rays[1].ori.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.ori.z), rays[1].ori.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.ori.x), rays[2].ori.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.ori.y), rays[2].ori.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.ori.z), rays[2].ori.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.ori.x), rays[3].ori.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.ori.y), rays[3].ori.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.ori.z), rays[3].ori.z);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.ori.x), rays[4].ori.x);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.ori.y), rays[4].ori.y);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.ori.z), rays[4].ori.z);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.ori.x), rays[5].ori.x);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.ori.y), rays[5].ori.y);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.ori.z), rays[5].ori.z);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.ori.x), rays[6].ori.x);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.ori.y), rays[6].ori.y);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.ori.z), rays[6].ori.z);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.ori.x), rays[7].ori.x);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.ori.y), rays[7].ori.y);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.ori.z), rays[7].ori.z);


        EXPECT_FLOAT_EQ(simd::get<0>(ray8.dir.x), rays[0].dir.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray8.dir.y), rays[0].dir.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray8.dir.z), rays[0].dir.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.dir.x), rays[1].dir.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.dir.y), rays[1].dir.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray8.dir.z), rays[1].dir.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.dir.x), rays[2].dir.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.dir.y), rays[2].dir.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray8.dir.z), rays[2].dir.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.dir.x), rays[3].dir.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.dir.y), rays[3].dir.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray8.dir.z), rays[3].dir.z);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.dir.x), rays[4].dir.x);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.dir.y), rays[4].dir.y);
        EXPECT_FLOAT_EQ(simd::get<4>(ray8.dir.z), rays[4].dir.z);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.dir.x), rays[5].dir.x);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.dir.y), rays[5].dir.y);
        EXPECT_FLOAT_EQ(simd::get<5>(ray8.dir.z), rays[5].dir.z);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.dir.x), rays[6].dir.x);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.dir.y), rays[6].dir.y);
        EXPECT_FLOAT_EQ(simd::get<6>(ray8.dir.z), rays[6].dir.z);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.dir.x), rays[7].dir.x);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.dir.y), rays[7].dir.y);
        EXPECT_FLOAT_EQ(simd::get<7>(ray8.dir.z), rays[7].dir.z);
    }

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX
}
