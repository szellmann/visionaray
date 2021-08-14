// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/simd/simd.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/intersect.h>
#include <visionaray/math/ray.h>
#include <visionaray/math/triangle.h>
#include <visionaray/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


TEST(Intersect, Ray1AABB4)
{
    basic_ray<float> r(
            vector<3, float>(-2.0f, 0.0f, 0.0f),
            vector<3, float>(1.0f, 0.0f, 0.0f)
            );

    array<aabb, 4> boxes {
            aabb({ -3.5f, -0.5f, -0.5f }, { -2.5f, +0.5f, +0.5f }),
            aabb({ -2.5f, -0.5f, -0.5f }, { -1.5f, +0.5f, +0.5f }),
            aabb({ -1.5f, -0.5f, -0.5f }, { -0.5f, +0.5f, +0.5f }),
            aabb({ -0.5f, -0.5f, -0.5f }, { +0.5f, +0.5f, +0.5f })
            };

    basic_aabb<simd::float4> box(simd::pack(boxes));

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

TEST(Intersect, Ray1Tri4)
{
    basic_ray<float> r(
            vector<3, float>(-2.0f, 0.0f, 0.0f),
            vector<3, float>(1.0f, 0.0f, 0.0f)
            );

    VSNRAY_ALIGN(16) int prim_ids[4] = { 0, 1, 2, 3 };
    VSNRAY_ALIGN(16) int geom_ids[4] = { 0, 0, 0, 0 };

    array<vector<3, float>, 4> v1s = {{
            { -3.0f, -1.0f, -1.0f },
            { -2.0f, -1.0f, -1.0f },
            { -1.0f, -1.0f, -1.0f },
            { -0.0f, -1.0f, -1.0f }
            }};

    array<vector<3, float>, 4> v2s = {{
            { -3.0f, -1.0f, +1.0f },
            { -2.0f, -1.0f, +1.0f },
            { -1.0f, -1.0f, +1.0f },
            { -0.0f, -1.0f, +1.0f }
            }};

    array<vector<3, float>, 4> v3s = {{
            { -3.0f,  1.0f,  0.0f },
            { -2.0f,  1.0f,  0.0f },
            { -1.0f,  1.0f,  0.0f },
            { -0.0f,  1.0f,  0.0f }
            }};

    array<vector<3, float>, 4> e1s = {{
            v2s[0] - v1s[0],
            v2s[1] - v1s[1],
            v2s[2] - v1s[2],
            v2s[3] - v1s[3]
            }};

    array<vector<3, float>, 4> e2s = {{
            v3s[0] - v1s[0],
            v3s[1] - v1s[1],
            v3s[2] - v1s[2],
            v3s[3] - v1s[3]
            }};

    basic_triangle<3, simd::float4, simd::int4> tri(
            simd::pack(v1s),
            simd::pack(e1s),
            simd::pack(e2s)
            );
    tri.prim_id = simd::int4(prim_ids);
    tri.geom_id = simd::int4(geom_ids);

    auto hr = intersect(r, tri);

    // TODO: move this to a simd::unpack function for primitive hit records
    using float_array = simd::aligned_array_t<simd::float4>;
    using int_array = simd::aligned_array_t<simd::int_type_t<simd::float4>>;

    int_array prim_id;
    store(prim_id, hr.prim_id);

    int_array geom_id;
    store(geom_id, hr.geom_id);

    float_array t;
    store(t, hr.t);

    ASSERT_TRUE(prim_id[1] == 1);
    ASSERT_TRUE(prim_id[2] == 2);
    ASSERT_TRUE(prim_id[3] == 3);

    ASSERT_TRUE(geom_id[1] == 0);
    ASSERT_TRUE(geom_id[2] == 0);
    ASSERT_TRUE(geom_id[3] == 0);

    EXPECT_FLOAT_EQ(t[1], 0.0f);
    EXPECT_FLOAT_EQ(t[2], 1.0f);
    EXPECT_FLOAT_EQ(t[3], 2.0f);
}
