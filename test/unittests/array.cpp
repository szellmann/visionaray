// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstring> // memcpy
#include <numeric>
#include <utility>

#include <visionaray/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test reverse iterators
//

TEST(Array, ReverseIt)
{
    array<int, 5> arr1;
    std::iota(arr1.begin(), arr1.end(), 0);

    // Test non-const iterators for writing
    array<int, 5> arr2;
    std::iota(arr2.rbegin(), arr2.rend(), 0);


    // Test const reverse iterators obtained implicitly through rbegin() and rend()
    auto it1 = arr1.rbegin();
    auto it2 = arr2.begin();
    for (; it1 != arr1.rend() && it2 != arr2.end(); ++it1, ++it2)
    {
        EXPECT_EQ(*it1, *it2);
    }

    // Test const reverse iterators obtained through crbegin() and crend()
    auto cit1 = arr1.crbegin();
    auto cit2 = arr2.cbegin();
    for (; cit1 != arr1.crend() && cit2 != arr2.cend(); ++cit1, ++cit2)
    {
        EXPECT_EQ(*cit1, *cit2);
    }
}


//-------------------------------------------------------------------------------------------------
// Test array::fill()
//

TEST(Array, Fill)
{
    const int N = 50;
    int value = 23;
    array<int, N> arr;
    arr.fill(23);

    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(arr[i], value);
    }
}


//-------------------------------------------------------------------------------------------------
// Test array::swap()
//

TEST(Array, Swap)
{
    const int N = 50;

    array<int, N> arr1;
    arr1.fill(23);

    array<int, N> arr2;
    arr2.fill(24);

    arr1.swap(arr2);

    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(arr1[i], 24);
        EXPECT_EQ(arr2[i], 23);
    }
}


//-------------------------------------------------------------------------------------------------
// Test interoperability with std::swap()
//

TEST(Array, StdSwap)
{
    const int N = 50;

    array<int, N> arr1;
    arr1.fill(23);

    array<int, N> arr2;
    arr2.fill(24);

    std::swap(arr1, arr2);

    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(arr1[i], 24);
        EXPECT_EQ(arr2[i], 23);
    }
}
