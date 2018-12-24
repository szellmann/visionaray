// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/aligned_vector.h>
#include <visionaray/array_ref.h>
#include <visionaray/bvh.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Helpers
//

using sphere_t   = basic_sphere<float>;
using triangle_t = basic_triangle<3, float>;

template <typename P>
using array_ref_bvh = index_bvh_t<array_ref<P>, aligned_vector<bvh_node, 32>, aligned_vector<unsigned, 32>>;


// generate some triangles --------------------------------

aligned_vector<triangle_t, 32> make_triangles()
{
    aligned_vector<triangle_t, 32> triangles;
    triangles.emplace_back(
            vec3(0.0f, 0.0f, 0.0f),
            vec3(1.0f, 0.0f, 0.0f),
            vec3(1.0f, 1.0f, 0.0f)
            );

    triangles.emplace_back(
            vec3(0.0f, 0.0f, 0.0f),
            vec3(1.0f, 1.0f, 0.0f),
            vec3(0.0f, 1.0f, 0.0f)
            );

    triangles.emplace_back(
            vec3(0.0f, 0.0f, 0.0f),
            vec3(1.0f, 0.0f, 0.0f),
            vec3(1.0f, 0.0f, 1.0f)
            );
    return triangles;
}

// generate some spheres ----------------------------------

aligned_vector<sphere_t, 32> make_spheres()
{
    aligned_vector<sphere_t, 32> spheres;
    spheres.emplace_back(vec3(0.0f, 0.0f, 0.0f), 1.0f);
    spheres.emplace_back(vec3(1.0f, 2.0f, 0.1f), 1000.0f);
    spheres.emplace_back(vec3(2.0f, 4.0f, 0.2f), 100000.0f);
    spheres.emplace_back(vec3(3.0f, 6.0f, 0.3f), 0.0001f);
    spheres.emplace_back(vec3(4.0f, 8.0f, 0.4f), 0.000000001f);
    return spheres;
}


//-------------------------------------------------------------------------------------------------
// Test build methods for several BVH types
//

// bvh ----------------------------------------------------

TEST(BVH, BuildBvh)
{
    binned_sah_builder builder;

    auto triangles = make_triangles();
    auto spheres   = make_spheres();

    auto triangle_bvh = builder.build<bvh<triangle_t>>(triangles.data(), triangles.size());
    auto sphere_bvh   = builder.build<bvh<sphere_t>>(spheres.data(), spheres.size());

    EXPECT_TRUE(triangle_bvh.nodes().size() > 0);
    EXPECT_TRUE(sphere_bvh.nodes().size()   > 0);

    EXPECT_TRUE(triangle_bvh.primitives().size() == triangles.size());
    EXPECT_TRUE(sphere_bvh.primitives().size()   == spheres.size());
}

// index bvh ----------------------------------------------

TEST(BVH, BuildIndexBvh)
{
    binned_sah_builder builder;

    auto triangles = make_triangles();
    auto spheres   = make_spheres();

    auto triangle_bvh = builder.build<index_bvh<triangle_t>>(triangles.data(), triangles.size());
    auto sphere_bvh   = builder.build<index_bvh<sphere_t>>(spheres.data(), spheres.size());

    EXPECT_TRUE(triangle_bvh.nodes().size() > 0);
    EXPECT_TRUE(sphere_bvh.nodes().size()   > 0);

    EXPECT_TRUE(triangle_bvh.primitives().size() == triangles.size());
    EXPECT_TRUE(sphere_bvh.primitives().size()   == spheres.size());
}

// index bvh w/ array_ref ---------------------------------

TEST(BVH, BuildArrayRefBvh)
{
    binned_sah_builder builder;

    auto triangles = make_triangles();
    auto spheres   = make_spheres();

    auto triangle_bvh = builder.build<array_ref_bvh<triangle_t>>(triangles.data(), triangles.size());
    auto sphere_bvh   = builder.build<array_ref_bvh<sphere_t>>(spheres.data(), spheres.size());

    EXPECT_TRUE(triangle_bvh.nodes().size() > 0);
    EXPECT_TRUE(sphere_bvh.nodes().size()   > 0);

    EXPECT_TRUE(triangle_bvh.primitives().size() == triangles.size());
    EXPECT_TRUE(sphere_bvh.primitives().size()   == spheres.size());
}
