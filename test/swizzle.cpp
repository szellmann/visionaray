// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vector>

#include <visionaray/math/math.h>
#include <visionaray/swizzle.h>

#include <gtest/gtest.h>

using namespace visionaray;

using unorm8_3 = vector<3, unorm<8>>;
using unorm8_4 = vector<4, unorm<8>>;


//-------------------------------------------------------------------------------------------------
// Test in-place swizzling
//

TEST(Swizzle, InPlace)
{
    // BGRA8 -> RGBA8

    std::vector<unorm8_4> rgba8{
            unorm8_4(0.00f, 0.33f, 0.66f, 1.00f),
            unorm8_4(0.25f, 0.50f, 0.75f, 1.00f)
            };

    auto rgba8_cpy = rgba8;

    swizzle(
            rgba8_cpy.data(),
            PF_RGBA8,
            PF_BGRA8,
            rgba8_cpy.size()
            );

    for (size_t i = 0; i < rgba8_cpy.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgba8_cpy[i].x, rgba8[i].z);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].y, rgba8[i].y);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].z, rgba8[i].x);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].w, rgba8[i].w);
    }


    // RGBA8 -> BGRA8

    swizzle(
            rgba8_cpy.data(),
            PF_BGRA8,
            PF_RGBA8,
            rgba8_cpy.size()
            );

    for (size_t i = 0; i < rgba8_cpy.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgba8_cpy[i].x, rgba8[i].x);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].y, rgba8[i].y);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].z, rgba8[i].z);
        EXPECT_FLOAT_EQ(rgba8_cpy[i].w, rgba8[i].w);
    }
}


//-------------------------------------------------------------------------------------------------
// Test conversions from RGB to RGBA
//

TEST(Swizzle, RGB2RGBA)
{
    // RGB8 -> RGBA8

    std::vector<unorm8_3> rgb8{
            unorm8_3(0.00f, 0.50f, 1.00f),
            unorm8_3(0.20f, 0.50f, 0.80f)
            };

    std::vector<unorm8_4> rgba8(rgb8.size());

    swizzle(
            rgba8.data(),
            PF_RGBA8,
            rgb8.data(),
            PF_RGB8,
            rgba8.size()
            );

    for (size_t i = 0; i < rgb8.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgba8[i].x, rgb8[i].x);
        EXPECT_FLOAT_EQ(rgba8[i].y, rgb8[i].y);
        EXPECT_FLOAT_EQ(rgba8[i].z, rgb8[i].z);
        EXPECT_FLOAT_EQ(rgba8[i].w, 1.0f);
    }
}


//-------------------------------------------------------------------------------------------------
// Test down conversions
//

TEST(Swizzle, Down)
{
    // RGB32F -> RGB8

    std::vector<vec3> rgb32f{
            vec3(0.0f, 0.5f, 1.0f),
            vec3(0.1f, 0.2f, 0.3f)
            };

    std::vector<unorm8_3> rgb8(rgb32f.size());

    swizzle(
            rgb8.data(),
            PF_RGB8,
            rgb32f.data(),
            PF_RGB32F,
            rgb8.size()
            );

    for (size_t i = 0; i < rgb32f.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgb8[i].x, static_cast<float>(unorm<8>(rgb32f[i].x)));
        EXPECT_FLOAT_EQ(rgb8[i].y, static_cast<float>(unorm<8>(rgb32f[i].y)));
        EXPECT_FLOAT_EQ(rgb8[i].z, static_cast<float>(unorm<8>(rgb32f[i].z)));
    }


    // RGBA32F -> RGBA8

    std::vector<vec4> rgba32f{
            vec4(0.0f, 0.2f, 0.4f, 0.6f),
            vec4(0.4f, 0.6f, 0.8f, 1.0f)
            };

    std::vector<unorm8_4> rgba8(rgba32f.size());

    swizzle(
            rgba8.data(),
            PF_RGBA8,
            rgba32f.data(),
            PF_RGBA32F,
            rgba8.size()
            );

    for (size_t i = 0; i < rgba32f.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgba8[i].x, static_cast<float>(unorm<8>(rgba32f[i].x)));
        EXPECT_FLOAT_EQ(rgba8[i].y, static_cast<float>(unorm<8>(rgba32f[i].y)));
        EXPECT_FLOAT_EQ(rgba8[i].z, static_cast<float>(unorm<8>(rgba32f[i].z)));
        EXPECT_FLOAT_EQ(rgba8[i].w, static_cast<float>(unorm<8>(rgba32f[i].w)));
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "fast";
    return RUN_ALL_TESTS();
}
