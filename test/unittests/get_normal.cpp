// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>
#include <visionaray/bvh.h>
#include <visionaray/get_normal.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test get_normal() for BVHs
//

TEST(GetNormal, BVH)
{
    using triangle_type = basic_triangle<3, float>;

    triangle_type triangles[2];

    triangles[0].v1 = vec3(-1.0f, -1.0f,  1.0f);
    triangles[0].e1 = vec3( 1.0f, -1.0f,  1.0f) - triangles[0].v1;
    triangles[0].e2 = vec3( 1.0f,  1.0f,  1.0f) - triangles[0].v1;
    triangles[0].prim_id = 0;
    triangles[0].geom_id = 0;

    triangles[1].v1 = vec3( 1.0f, -1.0f, -1.0f);
    triangles[1].e1 = vec3(-1.0f, -1.0f, -1.0f) - triangles[1].v1;
    triangles[1].e2 = vec3(-1.0f,  1.0f, -1.0f) - triangles[2].v1;
    triangles[1].prim_id = 1;
    triangles[1].geom_id = 1;


    auto bvh = build<index_bvh<triangle_type>>(triangles, 2);

    ray r;
    r.ori = vec3(0.5f, -0.5f, 2.0f);
    r.dir = normalize( vec3(0.0f, 0.0f, -1.0f) );
    auto hr = intersect(r, bvh);

    // Make some basic assumptions about the hit record
    EXPECT_TRUE(hr.hit);
    EXPECT_FLOAT_EQ(hr.t, 1.0f);
    EXPECT_EQ(hr.prim_id, 0);

    // Now test get_normal()
    auto n1 = get_normal(hr, bvh);
    auto n2 = normalize( cross(triangles[hr.prim_id].e1, triangles[hr.prim_id].e2) );

    EXPECT_FLOAT_EQ(n1.x, n2.x);
    EXPECT_FLOAT_EQ(n1.y, n2.y);
    EXPECT_FLOAT_EQ(n1.z, n2.z);
}
