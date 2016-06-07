// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstring> // memcpy
#include <utility>

#include <visionaray/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


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
