// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/intersect.h>
#include <visionaray/math/ray.h>
#include <visionaray/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


TEST(Intersect, RayAABB4)
{
    basic_ray<float> r(
            vector<3, float>(-2.0f, 0.0f, 0.0f),
            vector<3, float>(1.0f, 0.0f, 0.0f)
            );

    array<vector<3, float>, 4> box_mins = {{
            { -3.5f, -0.5f, -0.5f },
            { -2.5f, -0.5f, -0.5f },
            { -1.5f, -0.5f, -0.5f },
            { -0.5f, -0.5f, -0.5f },
            }};
    array<vector<3, float>, 4> box_maxs = {{
            { -2.5f, +0.5f, +0.5f },
            { -1.5f, +0.5f, +0.5f },
            { -0.5f, +0.5f, +0.5f },
            { +0.5f, +0.5f, +0.5f },
            }};

    basic_aabb<simd::float4> box(simd::pack(box_mins), simd::pack(box_maxs));

    auto hr = intersect(r, box);

    // TODO: move this to a simd::unpack function for AABB hit records
    using float_array = simd::aligned_array_t<simd::float4>;
    using int_array = simd::aligned_array_t<simd::int_type_t<simd::float4>>;

    int_array hit;
    store(hit, convert_to_int(hr.hit));

    float_array tnear;
    store(tnear, hr.tnear);

    float_array tfar;
    store(tfar, hr.tfar);

    // ASSERT_FALSE(hit[0]); // TODO: bug in intersect(), hit is also true if hit < r.tmin!
    ASSERT_TRUE(hit[1]);
    ASSERT_TRUE(hit[2]);
    ASSERT_TRUE(hit[3]);

    EXPECT_FLOAT_EQ(tnear[2], 0.5f);
    EXPECT_FLOAT_EQ(tnear[3], 1.5f);

    EXPECT_FLOAT_EQ(tfar[1], 0.5f);
    EXPECT_FLOAT_EQ(tfar[2], 1.5f);
    EXPECT_FLOAT_EQ(tfar[3], 2.5f);
}
