// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

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

        // pack with std::array
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

        // directly pack four rays
        auto ray44 = simd::pack(
                rays[0],
                rays[1],
                rays[2],
                rays[3]
                );

        EXPECT_FLOAT_EQ(simd::get<0>(ray44.ori.x), rays[0].ori.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray44.ori.y), rays[0].ori.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray44.ori.z), rays[0].ori.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.ori.x), rays[1].ori.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.ori.y), rays[1].ori.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.ori.z), rays[1].ori.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.ori.x), rays[2].ori.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.ori.y), rays[2].ori.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.ori.z), rays[2].ori.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.ori.x), rays[3].ori.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.ori.y), rays[3].ori.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.ori.z), rays[3].ori.z);

        EXPECT_FLOAT_EQ(simd::get<0>(ray44.dir.x), rays[0].dir.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray44.dir.y), rays[0].dir.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray44.dir.z), rays[0].dir.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.dir.x), rays[1].dir.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.dir.y), rays[1].dir.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray44.dir.z), rays[1].dir.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.dir.x), rays[2].dir.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.dir.y), rays[2].dir.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray44.dir.z), rays[2].dir.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.dir.x), rays[3].dir.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.dir.y), rays[3].dir.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray44.dir.z), rays[3].dir.z);
    }

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

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

        // directly pack eight rays
        auto ray88 = simd::pack(
                rays[0],
                rays[1],
                rays[2],
                rays[3],
                rays[4],
                rays[5],
                rays[6],
                rays[7]
                );

        EXPECT_FLOAT_EQ(simd::get<0>(ray88.ori.x), rays[0].ori.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray88.ori.y), rays[0].ori.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray88.ori.z), rays[0].ori.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.ori.x), rays[1].ori.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.ori.y), rays[1].ori.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.ori.z), rays[1].ori.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.ori.x), rays[2].ori.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.ori.y), rays[2].ori.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.ori.z), rays[2].ori.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.ori.x), rays[3].ori.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.ori.y), rays[3].ori.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.ori.z), rays[3].ori.z);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.ori.x), rays[4].ori.x);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.ori.y), rays[4].ori.y);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.ori.z), rays[4].ori.z);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.ori.x), rays[5].ori.x);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.ori.y), rays[5].ori.y);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.ori.z), rays[5].ori.z);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.ori.x), rays[6].ori.x);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.ori.y), rays[6].ori.y);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.ori.z), rays[6].ori.z);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.ori.x), rays[7].ori.x);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.ori.y), rays[7].ori.y);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.ori.z), rays[7].ori.z);


        EXPECT_FLOAT_EQ(simd::get<0>(ray88.dir.x), rays[0].dir.x);
        EXPECT_FLOAT_EQ(simd::get<0>(ray88.dir.y), rays[0].dir.y);
        EXPECT_FLOAT_EQ(simd::get<0>(ray88.dir.z), rays[0].dir.z);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.dir.x), rays[1].dir.x);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.dir.y), rays[1].dir.y);
        EXPECT_FLOAT_EQ(simd::get<1>(ray88.dir.z), rays[1].dir.z);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.dir.x), rays[2].dir.x);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.dir.y), rays[2].dir.y);
        EXPECT_FLOAT_EQ(simd::get<2>(ray88.dir.z), rays[2].dir.z);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.dir.x), rays[3].dir.x);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.dir.y), rays[3].dir.y);
        EXPECT_FLOAT_EQ(simd::get<3>(ray88.dir.z), rays[3].dir.z);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.dir.x), rays[4].dir.x);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.dir.y), rays[4].dir.y);
        EXPECT_FLOAT_EQ(simd::get<4>(ray88.dir.z), rays[4].dir.z);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.dir.x), rays[5].dir.x);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.dir.y), rays[5].dir.y);
        EXPECT_FLOAT_EQ(simd::get<5>(ray88.dir.z), rays[5].dir.z);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.dir.x), rays[6].dir.x);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.dir.y), rays[6].dir.y);
        EXPECT_FLOAT_EQ(simd::get<6>(ray88.dir.z), rays[6].dir.z);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.dir.x), rays[7].dir.x);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.dir.y), rays[7].dir.y);
        EXPECT_FLOAT_EQ(simd::get<7>(ray88.dir.z), rays[7].dir.z);
    }

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
}


TEST(Ray, SIMDUnpack)
{
    // SSE ------------------------------------------------

    {
        basic_ray<simd::float4> ray4(
            vector<3, simd::float4>(
                    simd::float4(0.0f, 1.0f, 2.0f, 3.0f),
                    simd::float4(0.3f, 1.3f, 2.3f, 3.3f),
                    simd::float4(0.5f, 1.5f, 2.5f, 3.5f)
                    ),
            vector<3, simd::float4>(
                    simd::float4(1.0f, 0.0f, 0.0f, 0.5f),
                    simd::float4(0.0f, 1.0f, 0.0f, 0.5f),
                    simd::float4(0.0f, 0.0f, 1.0f, 0.0f)
                    )
            );

        auto rays = unpack(ray4);

        EXPECT_FLOAT_EQ(rays[0].ori.x, simd::get<0>(ray4.ori.x));
        EXPECT_FLOAT_EQ(rays[0].ori.y, simd::get<0>(ray4.ori.y));
        EXPECT_FLOAT_EQ(rays[0].ori.z, simd::get<0>(ray4.ori.z));
        EXPECT_FLOAT_EQ(rays[1].ori.x, simd::get<1>(ray4.ori.x));
        EXPECT_FLOAT_EQ(rays[1].ori.y, simd::get<1>(ray4.ori.y));
        EXPECT_FLOAT_EQ(rays[1].ori.z, simd::get<1>(ray4.ori.z));
        EXPECT_FLOAT_EQ(rays[2].ori.x, simd::get<2>(ray4.ori.x));
        EXPECT_FLOAT_EQ(rays[2].ori.y, simd::get<2>(ray4.ori.y));
        EXPECT_FLOAT_EQ(rays[2].ori.z, simd::get<2>(ray4.ori.z));
        EXPECT_FLOAT_EQ(rays[3].ori.x, simd::get<3>(ray4.ori.x));
        EXPECT_FLOAT_EQ(rays[3].ori.y, simd::get<3>(ray4.ori.y));
        EXPECT_FLOAT_EQ(rays[3].ori.z, simd::get<3>(ray4.ori.z));

        EXPECT_FLOAT_EQ(rays[0].dir.x, simd::get<0>(ray4.dir.x));
        EXPECT_FLOAT_EQ(rays[0].dir.y, simd::get<0>(ray4.dir.y));
        EXPECT_FLOAT_EQ(rays[0].dir.z, simd::get<0>(ray4.dir.z));
        EXPECT_FLOAT_EQ(rays[1].dir.x, simd::get<1>(ray4.dir.x));
        EXPECT_FLOAT_EQ(rays[1].dir.y, simd::get<1>(ray4.dir.y));
        EXPECT_FLOAT_EQ(rays[1].dir.z, simd::get<1>(ray4.dir.z));
        EXPECT_FLOAT_EQ(rays[2].dir.x, simd::get<2>(ray4.dir.x));
        EXPECT_FLOAT_EQ(rays[2].dir.y, simd::get<2>(ray4.dir.y));
        EXPECT_FLOAT_EQ(rays[2].dir.z, simd::get<2>(ray4.dir.z));
        EXPECT_FLOAT_EQ(rays[3].dir.x, simd::get<3>(ray4.dir.x));
        EXPECT_FLOAT_EQ(rays[3].dir.y, simd::get<3>(ray4.dir.y));
        EXPECT_FLOAT_EQ(rays[3].dir.z, simd::get<3>(ray4.dir.z));
    }

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)

    // AVX ------------------------------------------------

    {
        basic_ray<simd::float8> ray8(
            vector<3, simd::float8>(
                    simd::float8(0.0f, 1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f, 7.0f),
                    simd::float8(0.3f, 1.3f, 2.3f, 3.3f,  4.3f, 5.3f, 6.3f, 7.3f),
                    simd::float8(0.5f, 1.5f, 2.5f, 3.5f,  4.5f, 5.5f, 6.5f, 7.5f)
                    ),
            vector<3, simd::float8>(
                    simd::float8(1.0f, 0.0f, 0.0f, 0.5f,  0.5f, 0.0f, 1.0f, 0.0f),
                    simd::float8(0.0f, 1.0f, 0.0f, 0.5f,  0.0f, 0.5f, 0.0f, 1.0f),
                    simd::float8(0.0f, 0.0f, 1.0f, 0.0f,  0.5f, 0.5f, 0.0f, 0.0f)
                    )
            );

        auto rays = unpack(ray8);

        EXPECT_FLOAT_EQ(rays[0].ori.x, simd::get<0>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[0].ori.y, simd::get<0>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[0].ori.z, simd::get<0>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[1].ori.x, simd::get<1>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[1].ori.y, simd::get<1>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[1].ori.z, simd::get<1>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[2].ori.x, simd::get<2>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[2].ori.y, simd::get<2>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[2].ori.z, simd::get<2>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[3].ori.x, simd::get<3>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[3].ori.y, simd::get<3>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[3].ori.z, simd::get<3>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[4].ori.x, simd::get<4>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[4].ori.y, simd::get<4>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[4].ori.z, simd::get<4>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[5].ori.x, simd::get<5>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[5].ori.y, simd::get<5>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[5].ori.z, simd::get<5>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[6].ori.x, simd::get<6>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[6].ori.y, simd::get<6>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[6].ori.z, simd::get<6>(ray8.ori.z));
        EXPECT_FLOAT_EQ(rays[7].ori.x, simd::get<7>(ray8.ori.x));
        EXPECT_FLOAT_EQ(rays[7].ori.y, simd::get<7>(ray8.ori.y));
        EXPECT_FLOAT_EQ(rays[7].ori.z, simd::get<7>(ray8.ori.z));

        EXPECT_FLOAT_EQ(rays[0].dir.x, simd::get<0>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[0].dir.y, simd::get<0>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[0].dir.z, simd::get<0>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[1].dir.x, simd::get<1>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[1].dir.y, simd::get<1>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[1].dir.z, simd::get<1>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[2].dir.x, simd::get<2>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[2].dir.y, simd::get<2>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[2].dir.z, simd::get<2>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[3].dir.x, simd::get<3>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[3].dir.y, simd::get<3>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[3].dir.z, simd::get<3>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[4].dir.x, simd::get<4>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[4].dir.y, simd::get<4>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[4].dir.z, simd::get<4>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[5].dir.x, simd::get<5>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[5].dir.y, simd::get<5>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[5].dir.z, simd::get<5>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[6].dir.x, simd::get<6>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[6].dir.y, simd::get<6>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[6].dir.z, simd::get<6>(ray8.dir.z));
        EXPECT_FLOAT_EQ(rays[7].dir.x, simd::get<7>(ray8.dir.x));
        EXPECT_FLOAT_EQ(rays[7].dir.y, simd::get<7>(ray8.dir.y));
        EXPECT_FLOAT_EQ(rays[7].dir.z, simd::get<7>(ray8.dir.z));
    }

#endif // VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
}
