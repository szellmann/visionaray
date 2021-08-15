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

    array<basic_triangle<3, float>, 4> tris;

    tris[0].prim_id = 0;
    tris[0].geom_id = 0;
    tris[0].v1 = vec3(-3.0f, -1.0f, -1.0f);
    tris[0].e1 = vec3(-3.0f, -1.0f, +1.0f) - tris[0].v1;
    tris[0].e2 = vec3(-3.0f,  1.0f,  0.0f) - tris[0].v1;

    tris[1].prim_id = 1;
    tris[1].geom_id = 0;
    tris[1].v1 = vec3(-2.0f, -1.0f, -1.0f);
    tris[1].e1 = vec3(-2.0f, -1.0f, +1.0f) - tris[1].v1;
    tris[1].e2 = vec3(-2.0f,  1.0f,  0.0f) - tris[1].v1;

    tris[2].prim_id = 2;
    tris[2].geom_id = 0;
    tris[2].v1 = vec3(-1.0f, -1.0f, -1.0f);
    tris[2].e1 = vec3(-1.0f, -1.0f, +1.0f) - tris[2].v1;
    tris[2].e2 = vec3(-1.0f,  1.0f,  0.0f) - tris[2].v1;

    tris[3].prim_id = 3;
    tris[3].geom_id = 0;
    tris[3].v1 = vec3(-0.0f, -1.0f, -1.0f);
    tris[3].e1 = vec3(-0.0f, -1.0f, +1.0f) - tris[3].v1;
    tris[3].e2 = vec3(-0.0f,  1.0f,  0.0f) - tris[3].v1;

    auto tri = simd::pack(tris);

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
