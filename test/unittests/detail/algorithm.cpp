// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <climits>
#include <functional>

#include <visionaray/detail/algorithm.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test insert_sorted()
//

TEST(Algorithm, InsertSorted)
{
    // Array of ints

    std::array<int, 4> arr = {{ INT_MAX, INT_MAX, INT_MAX, INT_MAX }};

    algo::insert_sorted(0, arr.begin(), arr.end(), std::less<int>());
    // 0, inf, inf, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], INT_MAX);
    EXPECT_EQ(arr[2], INT_MAX);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(10, arr.begin(), arr.end(), std::less<int>());
    // 0, 10, inf, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], INT_MAX);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 10, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 10);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 10
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 10);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 4
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);

    algo::insert_sorted(10, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 4 (10 is not inserted!)
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);

    algo::insert_sorted(-INT_MAX, arr.begin(), arr.end(), std::less<int>());
    // -inf, 0, 4, 4
    EXPECT_EQ(arr[0], -INT_MAX);
    EXPECT_EQ(arr[1], 0);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);
}
