// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <vector>

#include <visionaray/math/forward.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/swizzle.h>

#include <gtest/gtest.h>

using namespace visionaray;

using unorm8_3  = vector<3, unorm< 8>>;
using unorm8_4  = vector<4, unorm< 8>>;
using unorm16_3 = vector<3, unorm<16>>;
using unorm16_4 = vector<4, unorm<16>>;


//-------------------------------------------------------------------------------------------------
// Test in-place swizzling
//

TEST(Swizzle, InPlace)
{
    // BGR8 -> RGB8

    std::vector<unorm8_3> rgb8{
            unorm8_3(0.00f, 0.33f, 0.66f),
            unorm8_3(0.25f, 0.50f, 0.75f)
            };

    auto rgb8_cpy = rgb8;

    swizzle(
            rgb8_cpy.data(),
            PF_RGB8,
            PF_BGR8,
            rgb8_cpy.size()
            );

    for (size_t i = 0; i < rgb8_cpy.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgb8_cpy[i].x, rgb8[i].z);
        EXPECT_FLOAT_EQ(rgb8_cpy[i].y, rgb8[i].y);
        EXPECT_FLOAT_EQ(rgb8_cpy[i].z, rgb8[i].x);
    }


    // RGBA8 -> BGRA8

    swizzle(
            rgb8_cpy.data(),
            PF_BGR8,
            PF_RGB8,
            rgb8_cpy.size()
            );

    for (size_t i = 0; i < rgb8_cpy.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgb8_cpy[i].x, rgb8[i].x);
        EXPECT_FLOAT_EQ(rgb8_cpy[i].y, rgb8[i].y);
        EXPECT_FLOAT_EQ(rgb8_cpy[i].z, rgb8[i].z);
    }


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
// Test conversions from RGBA to RGB
//

TEST(Swizzle, RGBA2RGB)
{
    // RGBA8 -> RGB8, unorm8

    std::vector<unorm8_4> rgba8{
            unorm8_4(0.2f, 0.4f, 0.6f, 0.5f),
            unorm8_4(0.1f, 0.2f, 0.3f, 1.0f)
            };

    std::vector<unorm8_3> rgb8(rgba8.size());

    // premultiplied alpha

    swizzle(
            rgb8.data(),
            PF_RGB8,
            rgba8.data(),
            PF_RGBA8,
            rgb8.size(),
            PremultiplyAlpha
            );

    for (size_t i = 0; i < rgba8.size(); ++i)
    {
        float alpha = static_cast<float>(rgba8[i].w);

        unorm<8> x( static_cast<float>(rgba8[i].x) * alpha );
        unorm<8> y( static_cast<float>(rgba8[i].y) * alpha );
        unorm<8> z( static_cast<float>(rgba8[i].z) * alpha );

        EXPECT_FLOAT_EQ(rgb8[i].x, x);
        EXPECT_FLOAT_EQ(rgb8[i].y, y);
        EXPECT_FLOAT_EQ(rgb8[i].z, z);
    }

    // truncated alpha

    swizzle(
            rgb8.data(),
            PF_RGB8,
            rgba8.data(),
            PF_RGBA8,
            rgb8.size(),
            TruncateAlpha
            );

    for (size_t i = 0; i < rgba8.size(); ++i)
    {
        EXPECT_FLOAT_EQ(rgb8[i].x, rgba8[i].x);
        EXPECT_FLOAT_EQ(rgb8[i].y, rgba8[i].y);
        EXPECT_FLOAT_EQ(rgb8[i].z, rgba8[i].z);
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
            rgba8.size(),
            AlphaIsOne
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
    // RGB16UI -> RGB8

    std::vector<unorm16_3> rgb16ui{
            unorm16_3(0.0f, 0.5f, 1.0f),
            unorm16_3(0.1f, 0.2f, 0.3f)
            };

    std::vector<unorm8_3> rgb8(rgb16ui.size());

    swizzle(
            rgb8.data(),
            PF_RGB8,
            rgb16ui.data(),
            PF_RGB16UI,
            rgb16ui.size()
            );

    for (size_t i = 0; i < rgb16ui.size(); ++i)
    {
        ASSERT_NEAR(rgb8[i].x, rgb16ui[i].x, 0.002f);
        ASSERT_NEAR(rgb8[i].y, rgb16ui[i].y, 0.002f);
        ASSERT_NEAR(rgb8[i].z, rgb16ui[i].z, 0.002f);
    }


    // RGB32F -> RGB8

    std::vector<vec3> rgb32f{
            vec3(0.0f, 0.5f, 1.0f),
            vec3(0.1f, 0.2f, 0.3f)
            };

    rgb8.resize(rgb32f.size());

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


    // RGBA16UI -> RGBA8

    std::vector<unorm16_4> rgba16ui{
            unorm16_4(0.0f, 0.2f, 0.4f, 0.6f),
            unorm16_4(0.4f, 0.6f, 0.8f, 1.0f)
            };

    std::vector<unorm8_4> rgba8(rgba16ui.size());

    swizzle(
            rgba8.data(),
            PF_RGBA8,
            rgba16ui.data(),
            PF_RGBA16UI,
            rgba8.size()
            );

    for (size_t i = 0; i < rgba16ui.size(); ++i)
    {
        ASSERT_NEAR(rgba8[i].x, rgba16ui[i].x, 0.002f);
        ASSERT_NEAR(rgba8[i].y, rgba16ui[i].y, 0.002f);
        ASSERT_NEAR(rgba8[i].z, rgba16ui[i].z, 0.002f);
        ASSERT_NEAR(rgba8[i].w, rgba16ui[i].w, 0.002f);
    }


    // RGBA32F -> RGBA8

    std::vector<vec4> rgba32f{
            vec4(0.0f, 0.2f, 0.4f, 0.6f),
            vec4(0.4f, 0.6f, 0.8f, 1.0f)
            };

    rgba8.resize(rgba32f.size());

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
