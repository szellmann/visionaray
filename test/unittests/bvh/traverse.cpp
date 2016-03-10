// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/bvh.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Build some BVHs
//

//-------------------------------------------------------------------------------------------------
//
//              0
//            /   \
//           1     2
//

bvh<basic_triangle<3, float>> build_tiny_bvh()
{
    bvh_node n0;
    n0.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n0.bbox.max     = vec3( 1.0f,  1.0f,  1.0f);
    n0.first_child  = 1;
    n0.num_prims    = 0;

    bvh_node n1;
    n1.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n1.bbox.max     = vec3( 0.0f,  1.0f,  1.0f);
    n1.first_prim   = 0;
    n1.num_prims    = 2;

    bvh_node n2;
    n2.bbox.min     = vec3( 0.0f, -1.0f, -1.0f);
    n2.bbox.max     = vec3( 1.0f,  1.0f,  1.0f);
    n2.first_prim   = 2;
    n2.num_prims    = 2;


    bvh<basic_triangle<3, float>> tree;

    tree.nodes().push_back(n0);
    tree.nodes().push_back(n1);
    tree.nodes().push_back(n2);

    return tree;
}


//-------------------------------------------------------------------------------------------------
//
//              0
//            /   \
//           1     2
//         /   \
//        3     4
//      /   \
//     5     6
//

bvh<basic_triangle<3, float>> build_medium_bvh()
{
    bvh_node n0;
    n0.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n0.bbox.max     = vec3( 1.0f,  1.0f,  1.0f);
    n0.first_child  = 1;
    n0.num_prims    = 0;

    bvh_node n1;
    n1.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n1.bbox.max     = vec3( 0.0f,  1.0f,  1.0f);
    n1.first_child  = 4;
    n1.num_prims    = 0;

    bvh_node n2;
    n2.bbox.min     = vec3( 0.0f, -1.0f, -1.0f);
    n2.bbox.max     = vec3( 1.0f,  1.0f,  1.0f);
    n2.first_prim   = 0;
    n2.num_prims    = 2;

    bvh_node n3;
    n3.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n3.bbox.max     = vec3(-0.5f,  1.0f,  1.0f);
    n3.first_child  = 6;
    n3.num_prims    = 0;

    bvh_node n4;
    n4.bbox.min     = vec3(-0.5f, -1.0f, -1.0f);
    n4.bbox.max     = vec3( 1.0f,  1.0f,  1.0f);
    n4.first_prim   = 2;
    n4.num_prims    = 2;

    bvh_node n5;
    n5.bbox.min     = vec3(-1.0f, -1.0f, -1.0f);
    n5.bbox.max     = vec3(-0.7f,  1.0f,  1.0f);
    n5.first_prim   = 4;
    n5.num_prims    = 2;

    bvh_node n6;
    n6.bbox.min     = vec3(-0.7f, -1.0f, -1.0f);
    n6.bbox.max     = vec3(-0.5f,  1.0f,  1.0f);
    n6.first_prim   = 4;
    n6.num_prims    = 2;


    bvh<basic_triangle<3, float>> tree;

    tree.nodes().push_back(n0);
    tree.nodes().push_back(n1);
    tree.nodes().push_back(n2);
    tree.nodes().push_back(n3);
    tree.nodes().push_back(n4);
    tree.nodes().push_back(n5);
    tree.nodes().push_back(n6);

    return tree;
}


//-------------------------------------------------------------------------------------------------
// Test parent traversal
//

TEST(BVH, TraverseParents)
{
    // test tiny bvh --------------------------------------

    auto tree = build_tiny_bvh();

    traverse_parents(
        tree,
        tree.node(1),
        [&](bvh_node n)
        {
            EXPECT_TRUE(n == tree.node(0));
        }
        );

    traverse_parents(
        tree,
        tree.node(2),
        [&](bvh_node n)
        {
            EXPECT_TRUE(n == tree.node(0));
        }
        );


    // test medium size bvh -------------------------------

    tree = build_medium_bvh();


    int count = 0;

    // 6 - 3 - 1 - 0

    traverse_parents(
        tree,
        tree.node(6),
        [&](bvh_node n)
        {
            EXPECT_TRUE(count >= 0 && count < 3);

            if (count == 0)
            {
                EXPECT_TRUE(n == tree.node(3));
            }
            else if (count == 1)
            {
                EXPECT_TRUE(n == tree.node(1));
            }
            else if (count == 2)
            {
                EXPECT_TRUE(n == tree.node(0));
            }
            ++count;
        }
        );


    count = 0;

    // 5 - 3 - 1 - 0

    traverse_parents(
        tree,
        tree.node(5),
        [&](bvh_node n)
        {
            EXPECT_TRUE(count >= 0 && count < 3);

            if (count == 0)
            {
                EXPECT_TRUE(n == tree.node(3));
            }
            else if (count == 1)
            {
                EXPECT_TRUE(n == tree.node(1));
            }
            else if (count == 2)
            {
                EXPECT_TRUE(n == tree.node(0));
            }
            ++count;
        }
        );


    count = 0;

    // 4 - 1 - 0

    traverse_parents(
        tree,
        tree.node(4),
        [&](bvh_node n)
        {
            EXPECT_TRUE(count >= 0 && count < 2);

            if (count == 0)
            {
                EXPECT_TRUE(n == tree.node(1));
            }
            else if (count == 1)
            {
                EXPECT_TRUE(n == tree.node(0));
            }
            ++count;
        }
        );


    count = 0;

    // 3 - 1 - 0

    traverse_parents(
        tree,
        tree.node(3),
        [&](bvh_node n)
        {
            EXPECT_TRUE(count >= 0 && count < 2);

            if (count == 0)
            {
                EXPECT_TRUE(n == tree.node(1));
            }
            else if (count == 1)
            {
                EXPECT_TRUE(n == tree.node(0));
            }
            ++count;
        }
        );
}
